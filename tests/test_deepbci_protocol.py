"""Tests for realtime/deepbci_protocol.py — MIProtocol and trial sequence generation."""
import pytest

from realtime.deepbci_protocol import (
    MIProtocol,
    DEFAULT_TIMING,
    PHASE_ORDER,
    PHASE_LABELS,
    generate_random_sequence,
)


# ── MIProtocol ──────────────────────────────────────────────────────────────

class TestMIProtocolInit:
    def test_default_timing(self):
        p = MIProtocol()
        assert p.timing == DEFAULT_TIMING
        assert p.tick_s == 0.05
        assert p.dataset == "deepbci"
        assert p.current_phase == "idle"
        assert p.trial_count == 0

    def test_custom_timing(self):
        custom = {"rest_pre": 1.0, "cue": 0.5, "imagery": 3.0, "rest_post": 1.0}
        p = MIProtocol(timing=custom)
        assert p.timing == custom

    def test_custom_tick(self):
        p = MIProtocol(tick_s=0.1)
        assert p.tick_s == 0.1

    def test_custom_dataset(self):
        p = MIProtocol(dataset="physionet_mi")
        assert p.dataset == "physionet_mi"

    def test_unknown_dataset_falls_back_to_deepbci(self):
        p = MIProtocol(dataset="nonexistent")
        # Falls back to deepbci label map
        assert "idle" in p._label_map


class TestMIProtocolProperties:
    def test_total_trial_duration(self):
        p = MIProtocol()
        expected = sum(DEFAULT_TIMING.values())
        assert p.total_trial_duration == expected

    def test_class_labels_deepbci(self):
        p = MIProtocol()
        labels = p.class_labels
        assert "idle" in labels
        assert "left_hand" in labels
        assert "right_hand" in labels

    def test_class_labels_physionet(self):
        p = MIProtocol(dataset="physionet_mi")
        labels = p.class_labels
        assert "rest" in labels
        assert "left_hand" in labels
        assert "right_hand" in labels


class TestMIProtocolRunTrial:
    def test_yields_all_phases(self):
        """run_trial generator yields events in correct phase order."""
        p = MIProtocol()
        phases_seen = []
        for phase, elapsed in p.run_trial("left_hand"):
            if phase not in phases_seen or phase != phases_seen[-1]:
                phases_seen.append(phase)
        # Phase order should be: rest_pre, cue, imagery, rest_post
        assert phases_seen == PHASE_ORDER

    def test_elapsed_increases_within_phase(self):
        p = MIProtocol()
        elapsed_values = []
        current_phase = None
        for phase, elapsed in p.run_trial("left_hand"):
            if phase != current_phase:
                current_phase = phase
                elapsed_values = []
            elapsed_values.append(elapsed)
            if len(elapsed_values) >= 3:
                assert elapsed_values[-1] >= elapsed_values[-2]

    def test_elapsed_stays_within_duration(self):
        p = MIProtocol()
        for phase, elapsed in p.run_trial("left_hand"):
            assert elapsed <= p.timing[phase] + p.tick_s + 1e-6

    def test_increments_trial_count(self):
        p = MIProtocol()
        # Consume the generator
        for _ in p.run_trial("left_hand"):
            pass
        assert p.trial_count == 1

    def test_ends_in_idle_phase(self):
        p = MIProtocol()
        for _ in p.run_trial("left_hand"):
            pass
        assert p.current_phase == "idle"

    def test_unknown_label_raises(self):
        p = MIProtocol()
        with pytest.raises(ValueError, match="Unknown label"):
            next(p.run_trial("nonexistent_label"))

    def test_all_valid_labels_work(self):
        p = MIProtocol()
        for label in p.class_labels:
            gen = p.run_trial(label)
            # Consume first few ticks to verify it starts
            for _ in range(3):
                try:
                    next(gen)
                except StopIteration:
                    pass


class TestMIProtocolRunBlock:
    def test_yields_label_phase_elapsed(self):
        p = MIProtocol()
        trials = ["left_hand", "right_hand"]
        outputs = []
        for item in p.run_block(trials, inter_trial_pause=0.01):
            outputs.append(item)
        assert len(outputs) > 0
        label, phase, elapsed = outputs[0]
        assert label == "left_hand"
        assert phase in PHASE_ORDER

    def test_runs_all_trials(self):
        p = MIProtocol()
        trials = ["left_hand", "right_hand", "idle"]
        # Count unique trials by tracking label changes
        labels_seen = set()
        for label, _, _ in p.run_block(trials, inter_trial_pause=0.01):
            labels_seen.add(label)
        assert labels_seen == set(trials)

    def test_single_trial_block(self):
        p = MIProtocol()
        count = sum(1 for _ in p.run_block(["left_hand"], inter_trial_pause=0.01))
        assert count > 0

    def test_empty_block(self):
        p = MIProtocol()
        count = sum(1 for _ in p.run_block([], inter_trial_pause=0.01))
        assert count == 0


class TestMIProtocolStatus:
    def test_returns_dict(self):
        p = MIProtocol()
        st = p.status()
        assert isinstance(st, dict)
        assert "phase" in st
        assert "phase_label" in st
        assert "trial_count" in st
        assert "timing" in st

    def test_reflects_current_state(self):
        p = MIProtocol()
        # Run half a trial
        gen = p.run_trial("left_hand")
        for _ in range(10):
            next(gen)
        st = p.status()
        assert st["phase"] == PHASE_ORDER[0]  # still in rest_pre
        gen.close()  # clean up generator


# ── generate_random_sequence ────────────────────────────────────────────────

class TestGenerateRandomSequence:
    def test_correct_length(self):
        seq = generate_random_sequence(n_trials=30)
        assert len(seq) == 30

    def test_balanced_classes(self):
        seq = generate_random_sequence(n_trials=30, classes=["a", "b", "c"])
        assert seq.count("a") == 10
        assert seq.count("b") == 10
        assert seq.count("c") == 10

    def test_not_divisible_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            generate_random_sequence(n_trials=10, classes=["a", "b", "c"])

    def test_custom_classes(self):
        seq = generate_random_sequence(n_trials=20, classes=["left", "right"], seed=1)
        assert seq.count("left") == 10
        assert seq.count("right") == 10

    def test_reproducible_with_seed(self):
        seq1 = generate_random_sequence(n_trials=30, seed=42)
        seq2 = generate_random_sequence(n_trials=30, seed=42)
        assert seq1 == seq2

    def test_different_seeds_differ(self):
        seq1 = generate_random_sequence(n_trials=60, seed=1)
        seq2 = generate_random_sequence(n_trials=60, seed=2)
        assert seq1 != seq2

    def test_default_classes(self):
        seq = generate_random_sequence(n_trials=30)
        assert "idle" in seq
        assert "left_hand" in seq
        assert "right_hand" in seq

    def test_single_class(self):
        seq = generate_random_sequence(n_trials=10, classes=["only"])
        assert seq == ["only"] * 10


# ── Constants ───────────────────────────────────────────────────────────────

class TestConstants:
    def test_phase_order(self):
        assert PHASE_ORDER == ["rest_pre", "cue", "imagery", "rest_post"]

    def test_default_timing_values(self):
        assert DEFAULT_TIMING["rest_pre"] == 2.0
        assert DEFAULT_TIMING["cue"] == 1.0
        assert DEFAULT_TIMING["imagery"] == 4.0
        assert DEFAULT_TIMING["rest_post"] == 2.0

    def test_phase_labels(self):
        assert PHASE_LABELS["rest_pre"] == "Rest"
        assert PHASE_LABELS["cue"] == "←  CUE  →"
        assert PHASE_LABELS["imagery"] == "IMAGINE"
        assert PHASE_LABELS["rest_post"] == "Rest"
