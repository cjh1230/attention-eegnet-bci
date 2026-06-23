"""
Motor Imagery experiment protocol controller.

Defines the trial timing and state machine for a standard MI paradigm:

    [Rest 2s] → [Cue 1s] → [Imagery 4s] → [Rest 2s]

Designed to be driven by the Streamlit dashboard or a CLI script.
Emits (phase, elapsed_s) tuples so the UI can render real-time cues.

Usage:
    from realtime.deepbci_protocol import MIProtocol

    protocol = MIProtocol()
    for phase, elapsed in protocol.run_trial("left_hand"):
        print(f"{phase}: {elapsed:.1f}s")
        # → rest: 0.0s, rest: 0.1s, ..., cue: 0.0s, ..., imagery: 0.0s, ...
"""

import time
from collections.abc import Generator
from typing import Optional

from datasets.label_mapping import LABEL_MAPS

# ── Default timing (seconds) ──────────────────────────────────────────
DEFAULT_TIMING: dict[str, float] = {
    "rest_pre": 2.0,   # baseline rest before cue
    "cue": 1.0,        # visual/auditory cue
    "imagery": 4.0,    # motor imagery period
    "rest_post": 2.0,  # inter-trial rest
}

# ── Phase order ───────────────────────────────────────────────────────
PHASE_ORDER = ["rest_pre", "cue", "imagery", "rest_post"]

# ── Friendly labels for dashboard display ─────────────────────────────
PHASE_LABELS: dict[str, str] = {
    "rest_pre": "Rest",
    "cue": "←  CUE  →",
    "imagery": "IMAGINE",
    "rest_post": "Rest",
}


class MIProtocol:
    """
    Motor Imagery trial protocol with configurable timing.

    Parameters
    ----------
    timing : dict[str, float], optional
        Phase durations in seconds. Defaults to DEFAULT_TIMING.
    tick_s : float
        Tick interval for the generator (seconds).
    dataset : str
        Dataset key for semantic label lookup.
    """

    def __init__(
        self,
        timing: Optional[dict[str, float]] = None,
        tick_s: float = 0.05,
        dataset: str = "deepbci",
    ):
        self.timing = timing or dict(DEFAULT_TIMING)
        self.tick_s = tick_s
        self.dataset = dataset
        self._label_map = LABEL_MAPS.get(dataset, LABEL_MAPS["deepbci"])

        self.current_phase: str = "idle"
        self.trial_count: int = 0

    # ── Properties ─────────────────────────────────────────────────

    @property
    def total_trial_duration(self) -> float:
        """Total duration of one trial in seconds."""
        return sum(self.timing.values())

    @property
    def class_labels(self) -> list[str]:
        """Ordered list of semantic class names."""
        return sorted(self._label_map.keys(), key=lambda k: self._label_map[k])

    # ── Trial generator ────────────────────────────────────────────

    def run_trial(self, label: str) -> Generator[tuple[str, float], None, None]:
        """
        Run one MI trial as a generator.

        Yields (phase, elapsed_s) at self.tick_s intervals.

        Parameters
        ----------
        label : str
            Semantic label for this trial (e.g. "left_hand", "right_hand", "idle").

        Yields
        ------
        phase : str
            Current phase: "rest_pre", "cue", "imagery", "rest_post"
        elapsed_s : float
            Time elapsed within the current phase.
        """
        class_id = self._label_map.get(label)
        if class_id is None:
            raise ValueError(
                f"Unknown label '{label}' for dataset '{self.dataset}'. "
                f"Valid: {self.class_labels}"
            )

        for phase in PHASE_ORDER:
            self.current_phase = phase
            duration = self.timing[phase]
            elapsed = 0.0

            while elapsed < duration:
                yield phase, elapsed
                time.sleep(self.tick_s)
                elapsed += self.tick_s

            # Emit the final tick at exactly the phase boundary
            yield phase, duration

        self.trial_count += 1
        self.current_phase = "idle"

    # ── Block runner (for automated data collection) ───────────────

    def run_block(
        self,
        trials: list[str],
        inter_trial_pause: float = 1.0,
    ) -> Generator[tuple[str, str, float], None, None]:
        """
        Run a block of trials.

        Yields (label, phase, elapsed_s) for each trial.

        Parameters
        ----------
        trials : list[str]
            List of semantic labels, e.g. ["left_hand", "right_hand", "idle"].
        inter_trial_pause : float
            Extra pause between trials (seconds).

        Yields
        ------
        label : str
        phase : str
        elapsed_s : float
        """
        for i, label in enumerate(trials):
            yield from ((label, phase, elapsed) for phase, elapsed in self.run_trial(label))
            if i < len(trials) - 1:
                time.sleep(inter_trial_pause)

    # ── Status ─────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current protocol status for dashboard display."""
        return {
            "phase": self.current_phase,
            "phase_label": PHASE_LABELS.get(self.current_phase, self.current_phase),
            "trial_count": self.trial_count,
            "timing": dict(self.timing),
            "tick_s": self.tick_s,
        }


# ── Pre-built trial sequences ─────────────────────────────────────────

def generate_random_sequence(
    n_trials: int = 30,
    classes: Optional[list[str]] = None,
    seed: int = 42,
) -> list[str]:
    """
    Generate a balanced, shuffled trial sequence.

    Parameters
    ----------
    n_trials : int
        Total number of trials (must be divisible by len(classes)).
    classes : list[str], optional
        Semantic labels. Default: ["idle", "left_hand", "right_hand"]
    seed : int

    Returns
    -------
    sequence : list[str]
    """
    import random

    if classes is None:
        classes = ["idle", "left_hand", "right_hand"]

    if n_trials % len(classes) != 0:
        raise ValueError(
            f"n_trials ({n_trials}) must be divisible by n_classes ({len(classes)})"
        )

    rng = random.Random(seed)
    per_class = n_trials // len(classes)
    seq = classes * per_class
    rng.shuffle(seq)
    return seq
