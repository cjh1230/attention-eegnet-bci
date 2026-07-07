"""
Class Activation Topography (CAT) for EEG model visualization.

Integrates GradCAM (from Song et al. EEG-Conformer, TNSRE 2023) with
MNE topomaps for 8ch motor imagery EEG. Produces per-class activation
topographies + temporal heatmaps for EEGNet, EEG Conformer, EEG-TCNet.

Reference:
    Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding
    and Visualization", IEEE TNSRE, 2023.
    Official repo: https://github.com/eeyhsong/EEG-Conformer

Usage:
    # Single subject with EEG Conformer checkpoint
    python scripts/visualize_cat.py --model eeg_conformer \\
        --checkpoint checkpoints/conformer_best.pt --subject 7

    # EEGNet with per-class comparison
    python scripts/visualize_cat.py --model eegnet \\
        --checkpoint checkpoints/eegnet_best.pt --subject 7

    # All subjects (batch)
    python scripts/visualize_cat.py --model eeg_conformer \\
        --checkpoint checkpoints/conformer_best.pt --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.eegnet_attn import create_model
from utils.config import MOTOR_CHANNELS_BCI4 as CH_NAMES

OUTPUT = Path("results/cat_maps")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# GradCAM for 1D EEG signals (adapted from Song et al. EEG-Conformer)
# ──────────────────────────────────────────────────────────────────────────

class _ActivationsAndGradients:
    """Hook-based extraction of intermediate activations and gradients."""

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for layer in target_layers:
            self.handles.append(layer.register_forward_hook(self._save_activation))
            # handle both old and new PyTorch hook APIs
            if hasattr(layer, "register_full_backward_hook"):
                self.handles.append(
                    layer.register_full_backward_hook(self._save_gradient))
            else:
                self.handles.append(
                    layer.register_backward_hook(self._save_gradient))

    def _save_activation(self, module, inp, out):
        act = out
        if self.reshape_transform is not None:
            act = self.reshape_transform(act)
        self.activations.append(act.cpu().detach())

    def _save_gradient(self, module, grad_in, grad_out):
        grad = grad_out[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for h in self.handles:
            h.remove()


class GradCAM:
    """Gradient-weighted Class Activation Mapping for 1D/2D signals.

    Parameters
    ----------
    model : nn.Module
        The trained model (eval mode).
    target_layers : list[nn.Module]
        Layers to compute CAM from (typically last conv or norm layer).
    reshape_transform : callable or None
        For Transformer models: rearrange (B, T, D) → (B, D, 1, T).
    use_cuda : bool
        Whether to move model to GPU.
    """

    def __init__(self, model, target_layers, reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        if use_cuda and torch.cuda.is_available():
            self.model = model.cuda()
        self.act_grad = _ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    @staticmethod
    def _cam_weights(grads):
        # grads: (B, C, H, W) → mean over spatial dims → (B, C, 1, 1)
        return grads.mean(dim=(2, 3), keepdim=True)

    @staticmethod
    def _loss(output, target_category):
        loss = 0.0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def _cam_image(self, activations, grads):
        weights = self._cam_weights(grads)            # (B, C, 1, 1)
        return (weights * activations).sum(dim=1)      # (B, H, W)

    def __call__(self, input_tensor, target_category=None):
        """Compute GradCAM.

        Parameters
        ----------
        input_tensor : Tensor (B, ...)
        target_category : int, list[int], or None
            Target class(es). None → use predicted class.

        Returns
        -------
        cam : np.ndarray, shape (B, H, W)
            Normalised [0, 1] activation map.
        """
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        output = self.act_grad(input_tensor)

        if target_category is None:
            target_category = output.argmax(dim=-1).cpu().tolist()

        self.model.zero_grad()
        loss = self._loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = [a for a in self.act_grad.activations]
        grads = [g for g in self.act_grad.gradients]

        # Aggregate over target layers (usually just one)
        cams = []
        for act, grad in zip(activations, grads):
            cam = self._cam_image(act, grad)          # (B, H, W)
            cam = F.relu(cam)                          # mute negatives
            cam = self._normalise(cam)                 # [0, 1] per sample
            cams.append(cam)
        cams = torch.stack(cams, dim=1).max(dim=1).values  # (B, H, W)
        return cams.cpu().numpy()

    @staticmethod
    def _normalise(cam):
        """Min-max normalise each sample in batch to [0, 1]."""
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        mins = cam_flat.min(dim=1, keepdim=True).values.view(B, 1, 1)
        maxs = cam_flat.max(dim=1, keepdim=True).values.view(B, 1, 1)
        denom = (maxs - mins).clamp_min(1e-7)
        return (cam - mins) / denom

    def release(self):
        self.act_grad.release()


# ──────────────────────────────────────────────────────────────────────────
# Model-specific target layer resolution
# ──────────────────────────────────────────────────────────────────────────

def _find_target_layer(model, model_type: str) -> nn.Module:
    """Heuristic to pick the best target layer for GradCAM per architecture."""
    # Walk modules and pick the last "meaningful" layer before pooling/FC
    named = dict(model.named_modules())

    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv1d, nn.LayerNorm, nn.BatchNorm2d)):
            candidates.append((name, mod))

    if not candidates:
        raise ValueError("No Conv2d/Conv1d/LayerNorm/BatchNorm2d layer found.")

    if model_type in ("eeg_conformer",):
        # Prefer final LayerNorm (ln_final) for transformer-based models
        for name, mod in reversed(candidates):
            if "ln_final" in name:
                return mod
        # Fallback: last LayerNorm
        for name, mod in reversed(candidates):
            if isinstance(mod, nn.LayerNorm):
                return mod

    if model_type in ("eeg_tcnet", "fb_tcnet"):
        # Last Conv1d in TCN block
        for name, mod in reversed(candidates):
            if isinstance(mod, nn.Conv1d):
                return mod

    # Default (EEGNet, EEGNet variants, FBCNet): last BatchNorm2d
    for name, mod in reversed(candidates):
        if isinstance(mod, nn.BatchNorm2d):
            return mod
    # Ultimate fallback
    return candidates[-1][1]


def _get_reshape_transform(model_type: str):
    """For Transformer models: (B, T, D) → (B, D, 1, T)."""
    if model_type in ("eeg_conformer",):
        def _xfm(tensor):
            # tensor: (B, T, D) → (B, D, 1, T)
            return tensor.transpose(1, 2).unsqueeze(2)
        return _xfm
    return None


# ──────────────────────────────────────────────────────────────────────────
# MNE Topomap helper
# ──────────────────────────────────────────────────────────────────────────

def _make_8ch_info():
    """Create MNE Info for the 8ch motor montage (standard 10-20 names)."""
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    # CH_NAMES = ["FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4"]
    info = mne.create_info(ch_names=list(CH_NAMES), sfreq=250., ch_types="eeg")
    info.set_montage(montage)
    return info


# ──────────────────────────────────────────────────────────────────────────
# Per-subject CAT visualisation
# ──────────────────────────────────────────────────────────────────────────

def visualise_subject(
    model: nn.Module,
    model_type: str,
    subject_id: int,
    X: np.ndarray,     # (N, C, T) float32
    y: np.ndarray,      # (N,) int
    device: str,
    class_names: list[str],
):
    """Compute and save CAT visualisations for one subject."""
    n_classes = len(np.unique(y))
    N = X.shape[0]

    # ── target layer & reshape ──
    target_layer = _find_target_layer(model, model_type)
    reshape_fn = _get_reshape_transform(model_type)
    print(f"  Target layer: {type(target_layer).__name__}", flush=True)
    if reshape_fn:
        print(f"  Reshape transform: enabled (transformer mode)", flush=True)

    cam = GradCAM(model, [target_layer], reshape_transform=reshape_fn,
                  use_cuda=(device != "cpu"))

    # ── Per-trial CAM ──
    all_cams = []
    model.eval()
    with torch.enable_grad():
        for i in range(N):
            x = torch.as_tensor(X[i:i+1], dtype=torch.float32, device=device)
            c = cam(x, target_category=int(y[i]))
            all_cams.append(c[0])  # (H, W) — typically (1, T')
    all_cams = np.stack(all_cams, axis=0)  # (N, H, W)

    # ── Data for topomap (channel-mean over time) ──
    # X shape: (N, C, T); CAM shape: (N, 1, T')
    # Interpolate CAM to same time length as input for element-wise product
    T_in = X.shape[2]
    T_cam = all_cams.shape[2]
    cam_up = np.array([np.interp(
        np.linspace(0, T_cam - 1, T_in),
        np.arange(T_cam),
        all_cams[i, 0, :]
    ) for i in range(N)])  # (N, T_in)

    subj_dir = OUTPUT / f"S{subject_id:02d}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    info = _make_8ch_info()

    # ── Per-class figures ──
    for cls_id in range(n_classes):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        mask = y == cls_id
        if mask.sum() == 0:
            continue

        X_cls = X[mask]          # (n_cls, C, T)
        cam_cls_up = cam_up[mask]  # (n_cls, T_in)

        # Mean across trials
        x_mean = X_cls.mean(axis=0)      # (C, T)
        cam_mean = cam_cls_up.mean(axis=0)  # (T_in,)
        hyb_mean = x_mean * cam_mean[np.newaxis, :]  # (C, T), CAM-weighted EEG

        # Channel means for topomap
        ch_raw = x_mean.mean(axis=1)     # (C,) — raw EEG mean per channel
        ch_hyb = hyb_mean.mean(axis=1)   # (C,) — CAM-weighted mean per channel

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"S{subject_id:02d} — {cls_name}  |  {model_type}",
                     fontsize=14, fontweight="bold")

        # Row 1: Topomaps
        # (a) Raw topography
        im1, _ = mne.viz.plot_topomap(ch_raw, info, axes=axes[0, 0],
                                       show=False, res=600, cmap="RdBu_r")
        axes[0, 0].set_title("Raw EEG Topography\n(mean over time)")

        # (b) CAM-weighted topography
        im2, _ = mne.viz.plot_topomap(ch_hyb, info, axes=axes[0, 1],
                                       show=False, res=600, cmap="RdBu_r")
        axes[0, 1].set_title("CAT Topography\n(CAM × EEG, mean over time)")

        # (c) CAM channel saliency (hyb − raw)
        ch_diff = ch_hyb - ch_raw
        vmax_c = max(abs(ch_diff).max(), 0.01)
        im3, _ = mne.viz.plot_topomap(ch_diff, info, axes=axes[0, 2],
                                       show=False, res=600, cmap="RdBu_r",
                                       vlim=(-vmax_c, vmax_c))
        axes[0, 2].set_title("Δ Topography\n(CAT − Raw)")

        # Row 2: Time-domain visualisations
        # (d) CAM temporal profile
        axes[1, 0].plot(cam_mean, color="tab:red", linewidth=2)
        axes[1, 0].set_xlabel("Time (samples)")
        axes[1, 0].set_ylabel("CAM activation")
        axes[1, 0].set_title("Mean CAM Temporal Profile")
        axes[1, 0].grid(True, alpha=0.3)

        # (e) Channel × Time CAM-weighted EEG heatmap
        im = axes[1, 1].imshow(hyb_mean, aspect="auto", cmap="RdBu_r",
                               origin="lower")
        axes[1, 1].set_yticks(range(len(CH_NAMES)))
        axes[1, 1].set_yticklabels(CH_NAMES, fontsize=8)
        axes[1, 1].set_xlabel("Time (samples)")
        axes[1, 1].set_ylabel("Channel")
        axes[1, 1].set_title("CAT-Weighted EEG\n(Channel × Time)")
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

        # (f) Raw EEG Channel × Time (for reference)
        im = axes[1, 2].imshow(x_mean, aspect="auto", cmap="RdBu_r",
                               origin="lower")
        axes[1, 2].set_yticks(range(len(CH_NAMES)))
        axes[1, 2].set_yticklabels(CH_NAMES, fontsize=8)
        axes[1, 2].set_xlabel("Time (samples)")
        axes[1, 2].set_ylabel("Channel")
        axes[1, 2].set_title("Raw EEG\n(Channel × Time)")
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

        plt.tight_layout()
        fname = subj_dir / f"cat_{cls_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {fname}", flush=True)

    # ── Cross-class comparison (binary: class 0 vs class 1 topomap diff) ──
    unique_labels = sorted(np.unique(y))
    if len(unique_labels) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"S{subject_id:02d} — Cross-Class CAT  |  {model_type}",
                     fontsize=14, fontweight="bold")

        all_topo_diffs = []
        for i, (ca, cb) in enumerate([(unique_labels[0], unique_labels[1])]):
            # Actually, let's show all pair-wise comparisons for clarity
            pass

        # Show class 0, class 1, and their difference as topomaps
        for idx, (cid, cname) in enumerate([
            (unique_labels[0], class_names[unique_labels[0]]),
            (unique_labels[1], class_names[unique_labels[1]]),
        ]):
            mask = y == cid
            X_c = X[mask]
            cam_c_up = cam_up[mask]
            x_m = X_c.mean(axis=0)
            cam_m = cam_c_up.mean(axis=0)
            hyb_m = x_m * cam_m[np.newaxis, :]
            ch_h = hyb_m.mean(axis=1)
            im, _ = mne.viz.plot_topomap(ch_h, info, axes=axes[idx],
                                          show=False, res=600, cmap="RdBu_r")
            axes[idx].set_title(f"CAT: {cname}")

        # Difference topomap
        ch_diffs = []
        for cid in [unique_labels[0], unique_labels[1]]:
            mask = y == cid
            X_c = X[mask]
            cam_c_up = cam_up[mask]
            x_m = X_c.mean(axis=0)
            cam_m = cam_c_up.mean(axis=0)
            ch_diffs.append((x_m * cam_m[np.newaxis, :]).mean(axis=1))
        ch_diff = ch_diffs[1] - ch_diffs[0]

        c0_name = class_names[unique_labels[0]]
        c1_name = class_names[unique_labels[1]]
        vmax = max(abs(ch_diff).max(), 0.01)
        im, _ = mne.viz.plot_topomap(ch_diff, info, axes=axes[2],
                                      show=False, res=600, cmap="RdBu_r",
                                      vlim=(-vmax, vmax))
        axes[2].set_title(f"Δ CAT: {c1_name} − {c0_name}")

        plt.tight_layout()
        fname = subj_dir / "cat_class_comparison.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {fname}", flush=True)

    cam.release()


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Class Activation Topography (CAT) for EEG models")
    parser.add_argument("--model", default="eeg_conformer",
                        choices=["eegnet", "eegnet_se", "eegnet_mhsa",
                                 "eegnet_temporal", "eegnet_spatiotemporal",
                                 "eeg_conformer", "eeg_tcnet", "fbcnet",
                                 "fb_tcnet", "fb_maa_eegnet"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated IDs, e.g. '7,29,9'")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", default="physionet_mi")
    parser.add_argument("--n_channels", type=int, default=8)
    parser.add_argument("--n_classes", type=int, default=2)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    print(f"Loading {args.model} from {args.checkpoint} ...", flush=True)
    model = create_model(args.model, n_channels=args.n_channels,
                         n_classes=args.n_classes)
    model.eval()

    # Warm-up (handles lazy classifiers like EEGNet)
    with torch.no_grad():
        dummy = torch.zeros(1, args.n_channels, 750, device=device)
        model(dummy)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Handle both raw state_dict and full checkpoint
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"  Model loaded.", flush=True)

    # ── Class names ──
    from datasets.label_mapping import class_names as get_class_names
    class_names = get_class_names(args.dataset)

    # ── Subjects ──
    if args.all:
        subject_ids = list(range(1, 31))
    elif args.subjects:
        subject_ids = [int(x.strip()) for x in args.subjects.split(",")]
    elif args.subject:
        subject_ids = [args.subject]
    else:
        print("ERROR: specify --subject, --subjects, or --all")
        sys.exit(1)

    # ── Load per-subject data ──
    from training.train_loso import load_per_subject_data
    all_subjects = load_per_subject_data(args.data_dir, 30)
    subj_map = {s["id"]: s for s in all_subjects}

    total_trials = 0
    for sid in subject_ids:
        s = subj_map.get(sid)
        if s is None:
            print(f"S{sid:02d}: not found, skipping")
            continue
        print(f"\n{'='*50}")
        print(f"S{sid:02d}: {s['X'].shape[0]} trials")
        print(f"{'='*50}")
        visualise_subject(model, args.model, sid, s["X"], s["y"],
                          device, class_names)
        total_trials += s["X"].shape[0]

    print(f"\nDone. {len(subject_ids)} subject(s), {total_trials} trials → {OUTPUT}")


if __name__ == "__main__":
    main()
