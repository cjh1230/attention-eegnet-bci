"""Regression tests for training/evaluation glue code."""

import sys

import numpy as np
import pytest
import torch
import torch.nn as nn


class _NoopLogger:
    def __init__(self, *args, **kwargs):
        pass

    def log(self, **kwargs):
        pass

    def close(self):
        pass


def test_fbcnet_filter_bank_runs_before_dataloader(monkeypatch):
    """FBCNet training should feed 4D filter-bank tensors into the model."""
    import models.fbcnet as fbcnet
    import training.train_eegnet as train_eegnet

    class StrictFBCModel(nn.Module):
        input_requires_filter_bank = True

        def __init__(self):
            super().__init__()
            self.n_channels = 8
            self.n_classes = 2
            self.n_bands = 1
            self.logits = nn.Parameter(torch.zeros(2))

        def forward(self, x):
            assert x.dim() == 4
            return self.logits.unsqueeze(0).expand(x.shape[0], -1)

    filter_calls = []

    def fake_apply_filter_bank(X):
        filter_calls.append(X.shape)
        return X[:, None, :, :]

    monkeypatch.setattr(
        train_eegnet,
        "create_model",
        lambda *args, **kwargs: StrictFBCModel(),
    )
    monkeypatch.setattr(fbcnet, "apply_filter_bank", fake_apply_filter_bank)
    monkeypatch.setattr(train_eegnet, "ExperimentLogger", _NoopLogger)

    X_train = np.random.randn(8, 8, 64).astype(np.float32)
    y_train = np.array([0, 1] * 4, dtype=np.int64)
    X_val = np.random.randn(4, 8, 64).astype(np.float32)
    y_val = np.array([0, 1, 0, 1], dtype=np.int64)

    _, ckpt = train_eegnet._train_one_run(
        X_train,
        y_train,
        X_val,
        y_val,
        model_type="fbcnet",
        n_channels=8,
        n_classes=2,
        device="cpu",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        label_smoothing=0.0,
        grad_clip=0.0,
        early_stop=0,
    )

    assert ckpt["model_type"] == "fbcnet"
    assert filter_calls == [(8, 8, 64), (4, 8, 64)]


def test_load_checkpoint_uses_saved_model_type(tmp_path):
    """Checkpoints from non-EEGNet models should reload into the same class."""
    from models.fbcnet import FBCNet
    from training.train_eegnet import load_checkpoint

    model = FBCNet(n_channels=8, n_classes=2)
    ckpt_path = tmp_path / "fbcnet_best.pt"
    torch.save(
        {
            "epoch": 3,
            "model_type": "fbcnet",
            "state_dict": model.state_dict(),
            "acc": 0.75,
            "config": {
                "n_channels": 8,
                "n_classes": 2,
                "n_times": 80,
            },
        },
        ckpt_path,
    )

    loaded = load_checkpoint(str(ckpt_path), device="cpu")

    assert isinstance(loaded, FBCNet)
    assert loaded.model_type == "fbcnet"


def test_fbcsp_validation_fits_filter_bank_on_train_only(monkeypatch):
    """FBCSP validation must not fit CSP filters with validation labels."""
    pytest.importorskip("mne")

    import features.csp as csp
    import training.train_baseline as train_baseline

    events = []

    class FakeFilterBankCSP:
        def __init__(self, n_components):
            self.n_components = n_components

        def fit(self, X, y):
            events.append(("fb_fit", X.copy(), y.copy()))
            return self

        def transform(self, X):
            events.append(("fb_transform", X.copy()))
            return np.ones((X.shape[0], self.n_components), dtype=np.float32)

    class FakeClassifier:
        def fit(self, X, y):
            events.append(("clf_fit", X.copy(), y.copy()))
            return self

        def score(self, X, y):
            events.append(("clf_score", X.copy(), y.copy()))
            return 0.5

    monkeypatch.setattr(csp, "FilterBankCSP", FakeFilterBankCSP)
    monkeypatch.setattr(csp, "_make_classifier", lambda name: FakeClassifier())

    X_train = np.zeros((4, 8, 64), dtype=np.float32)
    y_train = np.array([0, 1, 0, 1], dtype=np.int64)
    X_val = np.ones((2, 8, 64), dtype=np.float32)
    y_val = np.array([1, 1], dtype=np.int64)

    train_baseline._eval_fbcsp_val(
        X_train,
        y_train,
        X_val,
        y_val,
        n_components=3,
        classifier="lda",
    )

    fit_events = [event for event in events if event[0] == "fb_fit"]
    assert len(fit_events) == 1
    np.testing.assert_array_equal(fit_events[0][1], X_train)
    np.testing.assert_array_equal(fit_events[0][2], y_train)


def test_loso_skip_train_loads_checkpoint_instead_of_training(monkeypatch, tmp_path):
    """--skip_train should load the checkpoint for each fold and never train."""
    import training.train_loso as train_loso

    calls = {"load": 0, "train": 0}
    subjects = [
        {
            "id": 1,
            "X": np.zeros((4, 8, 64), dtype=np.float32),
            "y": np.array([0, 1, 0, 1], dtype=np.int64),
        },
        {
            "id": 2,
            "X": np.ones((4, 8, 64), dtype=np.float32),
            "y": np.array([0, 1, 0, 1], dtype=np.int64),
        },
    ]

    class FakeModel(nn.Module):
        model_type = "eegnet"

    def fake_load_checkpoint(path, device):
        calls["load"] += 1
        return FakeModel()

    def fail_train(*args, **kwargs):
        calls["train"] += 1
        raise AssertionError("train_on_subjects should not run in skip_train mode")

    def fake_eval(model, subject, device):
        return {
            "accuracy": 0.5,
            "f1_macro": 0.5,
            "kappa": 0.0,
            "n_trials": len(subject["y"]),
            "per_class_recall": {0: 1.0, 1: 0.0},
            "per_class_specificity": {0: 0.0, 1: 1.0},
            "per_class_f1": {0: 0.67, 1: 0.0},
        }

    monkeypatch.setattr(train_loso, "load_per_subject_data", lambda *args: subjects)
    monkeypatch.setattr(train_loso, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(train_loso, "train_on_subjects", fail_train)
    monkeypatch.setattr(train_loso, "evaluate_on_subject", fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_loso.py",
            "--data_dir",
            "unused",
            "--n_subjects",
            "2",
            "--skip_train",
            "--checkpoint",
            "fake.pt",
            "--output_dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )

    train_loso.main()

    assert calls == {"load": 2, "train": 0}
    assert (tmp_path / "loso_eegnet_seed42_summary.json").exists()
