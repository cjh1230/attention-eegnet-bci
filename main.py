"""
BCI Project — Single entry point.

Usage:
    python main.py setup           # Create conda env and install deps
    python main.py preprocess      # Run MNE pipeline on data/raw/
    python main.py baseline        # CSP+SVM baseline
    python main.py train           # Train EEGNet
    python main.py ablation        # Ablation study (EEGNet vs +Attn)
    python main.py demo            # Real-time demo (simulated stream)
    python main.py dashboard       # Streamlit dashboard

Quick start (first time):
    conda env create -f environment.yml
    conda activate bci
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_py(file: str, *extra_args):
    """Run a Python script from the project root."""
    args = [sys.executable, str(ROOT / file)] + list(extra_args)
    return subprocess.run(args, cwd=str(ROOT)).returncode


def cmd_setup():
    print("Creating conda environment 'bci'...")
    subprocess.run(
        ["conda", "env", "create", "-f", str(ROOT / "environment.yml")],
        check=False,
    )
    print("\n✅ Done. Run: conda activate bci")


def cmd_preprocess():
    run_py("preprocessing/run_mne_pipeline.py", *sys.argv[2:])


def cmd_baseline():
    run_py("training/train_baseline.py", *sys.argv[2:])


def cmd_train():
    run_py("training/train_eegnet.py", *sys.argv[2:])


def cmd_ablation():
    run_py("training/train_ablation.py", *sys.argv[2:])


def cmd_demo():
    print("Starting BCI real-time demo (simulated stream)...")
    from realtime.buffer import RingBuffer
    from realtime.stream import DummyStream
    from realtime.inference import MIInference
    import torch
    from models.eegnet import EEGNet

    model = EEGNet(n_channels=N_CHANNELS, n_classes=3)
    model.eval()

    buffer = RingBuffer()
    stream = DummyStream()
    inference = MIInference(model, buffer, device="cpu")

    stream.open()
    print("Stream opened. Running inference loop (Ctrl+C to stop)...\n")

    try:
        for step in range(100):
            chunk = stream.read_chunk()
            buffer.push(chunk)
            class_id, conf = inference.predict()
            labels = ["[IDLE]", "[LEFT]", "[RIGHT]"]
            bar = "#" * int(conf * 20)
            print(f"  [{step:3d}] → {labels[class_id]:6s} | conf={conf:.2f} | {bar}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stream.close()


def cmd_dashboard():
    """Launch Streamlit dashboard."""
    subprocess.run([
        sys.executable,
        "-m", "streamlit", "run", str(ROOT / "ui" / "dashboard.py"),
    ], cwd=str(ROOT))


COMMANDS = {
    "setup": cmd_setup,
    "preprocess": cmd_preprocess,
    "baseline": cmd_baseline,
    "train": cmd_train,
    "ablation": cmd_ablation,
    "demo": cmd_demo,
    "dashboard": cmd_dashboard,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>")
        print("Commands:", ", ".join(sorted(COMMANDS)))
        return

    cmd = sys.argv[1]
    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print("Available:", ", ".join(sorted(COMMANDS)))


if __name__ == "__main__":
    main()
