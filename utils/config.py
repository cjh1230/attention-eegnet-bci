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
# 16 motor-cortex channels (10-10 system) — motor imagery sweet spot
MOTOR_CHANNELS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp3.", "Cp4.",
]

# --- MI Paradigm ---
FREQ_BANDS = {
    "mu": (8, 13),
    "beta": (13, 30),
    "full": (8, 30),
}
EVENT_IDS = {
    "rest": 0,
    "left_hand": 1,
    "right_hand": 2,
}
T_MIN, T_MAX = -0.5, 2.5  # epoch window around cue (s)

# --- Real-time ---
BUFFER_WINDOW = 2.0  # seconds
PREDICT_INTERVAL = 0.125  # 8 Hz inference rate

# --- Training ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 300
