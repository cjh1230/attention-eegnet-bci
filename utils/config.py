"""
Global configuration constants for the BCI project.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SUBJECTS = ROOT / "data" / "subjects"
CHECKPOINT_DIR = ROOT / "checkpoints"

# --- EEG Hardware ---
N_CHANNELS = 16
SFREQ = 250  # Hz (DeepBCI default; adjust per device)
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6",
]

# --- MI Paradigm ---
FREQ_BANDS = {
    "mu": (8, 13),
    "beta": (13, 30),
    "full": (8, 30),
}
EVENT_IDS = {
    "left_hand": 0,
    "right_hand": 1,
    "idle": 2,
}
T_MIN, T_MAX = -0.5, 2.5  # epoch window around cue (s)

# --- Real-time ---
BUFFER_WINDOW = 2.0  # seconds
PREDICT_INTERVAL = 0.125  # 8 Hz inference rate

# --- Training ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 300
