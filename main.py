"""
BCI Project — Single entry point.

Usage:
    python main.py setup           # Create conda env and install deps
    python main.py preprocess      # Run MNE pipeline (--dataset physionet_mi)
    python main.py baseline        # CSP+SVM baseline
    python main.py train           # Train EEGNet
    python main.py ablation        # Ablation study (EEGNet vs +Attn)
    python main.py loso            # LOSO cross-validation (gold-standard BCI eval)
    python main.py demo            # Real-time demo (simulated stream)
    python main.py dashboard       # Streamlit dashboard
    python main.py metadata        # Export dataset metadata to JSON
    python main.py subjectwise     # Subject-wise eval from checkpoint
    python main.py record          # DeepBCI data collection (interactive)
    python main.py run_all         # One-command: preprocess → train → LOSO → export
    python main.py export          # Generate competition Excel report
    python main.py figures         # Generate report figures (PNG)

Quick start (first time):
    conda env create -f environment.yml
    conda activate bci
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

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


def cmd_loso():
    run_py("training/train_loso.py", *sys.argv[2:])


def cmd_demo():
    print("Starting BCI real-time demo (simulated stream)...")
    from realtime.buffer import RingBuffer
    from realtime.stream import DummyStream
    from realtime.inference import MIInference
    import torch
    from models.eegnet import EEGNet
    from utils.config import N_CHANNELS

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


def cmd_metadata():
    """Export dataset metadata to JSON."""
    from datasets.metadata import export_metadata
    output = sys.argv[2] if len(sys.argv) > 2 else "results/metadata.json"
    export_metadata(output)


def cmd_subjectwise():
    """Subject-wise evaluation from a trained checkpoint."""
    run_py("training/evaluate_subjectwise.py", *sys.argv[2:])


def cmd_deepbci_record():
    """Launch DeepBCI recorder (interactive data collection)."""
    print("DeepBCI Recorder — interactive data collection")
    print("─" * 50)
    from realtime.deepbci_recorder import DeepBCIRecorder

    subject_id = int(input("Subject ID: ") or "1")
    notes = input("Notes (optional): ") or ""

    recorder = DeepBCIRecorder()
    session_dir = recorder.start_session(subject_id, notes=notes)
    print(f"\nSession directory: {session_dir}")
    print("Recording... (press Ctrl+C to stop)")
    try:
        import time
        t0 = time.time()
        while True:
            chunk = np.random.randn(8, 31).astype(np.float32)  # dummy chunk
            t = time.time() - t0
            recorder.record_chunk(chunk, t)
            time.sleep(0.125)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recorder.end_session()


def cmd_run_all():
    """Run all experiments (preprocess → train → LOSO → export)."""
    run_py("scripts/run_all_experiments.py", *sys.argv[2:])


def cmd_export():
    """Export competition-format Excel report."""
    run_py("scripts/export_competition_excel.py", *sys.argv[2:])


def cmd_figures():
    """Generate report figures from results/."""
    run_py("scripts/make_report_figures.py", *sys.argv[2:])


COMMANDS = {
    "setup": cmd_setup,
    "preprocess": cmd_preprocess,
    "baseline": cmd_baseline,
    "train": cmd_train,
    "ablation": cmd_ablation,
    "loso": cmd_loso,
    "demo": cmd_demo,
    "dashboard": cmd_dashboard,
    "metadata": cmd_metadata,
    "subjectwise": cmd_subjectwise,
    "record": cmd_deepbci_record,
    "run_all": cmd_run_all,
    "export": cmd_export,
    "figures": cmd_figures,
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
