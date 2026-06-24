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
    """Real-time BCI demo — configurable source, model, and gating."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time BCI demo (simulated or file-replay stream)"
    )
    parser.add_argument(
        "--source", choices=["dummy", "replay"], default="dummy",
        help="Stream source type (default: dummy)",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to X.npy for --source replay",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Path to y.npy for --source replay (optional)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to trained .pt checkpoint",
    )
    parser.add_argument(
        "--gating", action="store_true",
        help="Enable idle confidence gating",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Confidence threshold for gating (default: 0.65)",
    )
    parser.add_argument(
        "--all-subjects", action="store_true",
        help="Run demo across all subj_* dirs under --data parent directory",
    )
    args = parser.parse_args(sys.argv[2:])

    import torch
    from realtime.buffer import RingBuffer
    from realtime.inference import MIInference, build_action_map
    from models.eegnet_attn import create_model
    from utils.config import N_CHANNELS

    # ── Multi-subject mode ──────────────────────────────────────
    if args.all_subjects:
        if args.source != "replay":
            print("ERROR: --all-subjects requires --source replay")
            sys.exit(1)
        # Determine data directory: if --data points to a .npy file, use its parent dir
        if args.data:
            data_dir = Path(args.data)
            if data_dir.suffix == ".npy":
                data_dir = data_dir.parent
        else:
            data_dir = Path("data/loso_binary")
        subj_paths = sorted(data_dir.glob("subj_*/X.npy"))
        if not subj_paths:
            print(f"No subj_*/X.npy found under {data_dir}")
            sys.exit(1)

        # Load model once
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            n_classes = cfg.get("n_classes", 3)
            n_times = cfg.get("n_times", 750)
            model_type = (
                "eegnet_spatiotemporal" if "spatiotemporal" in args.checkpoint
                else "eegnet"
            )
            model = create_model(
                model_type, n_channels=N_CHANNELS, n_classes=n_classes,
                F1=cfg.get("F1", 8), D=cfg.get("D", 2),
                F2=cfg.get("F2", 16), dropout=cfg.get("dropout", 0.5),
            )
            model.eval()
            with torch.no_grad():
                model(torch.zeros(1, N_CHANNELS, n_times))
            model.load_state_dict(ckpt["state_dict"])
        else:
            model = create_model("eegnet", n_channels=N_CHANNELS, n_classes=3)
            n_classes = 3
        model.eval()

        # Build action map
        action_map = build_action_map(n_classes, "physionet_mi")

        print(f"Multi-subject demo: {len(subj_paths)} subjects")
        print(f"Model: {args.checkpoint or '(untrained)'}, {n_classes} classes\n")

        accuracies = {}
        for subj_path in subj_paths:
            subj_name = subj_path.parent.name
            labels_path = subj_path.parent / "y.npy"
            from realtime.file_replay import FileReplaySource
            source = FileReplaySource(
                data_path=str(subj_path),
                labels_path=str(labels_path) if labels_path.exists() else None,
                trial_mode=True,
            )
            source.open()

            correct = 0
            total = 0
            while not source.exhausted:
                chunk = source.read_chunk()
                if source.exhausted:
                    break
                tensor = torch.from_numpy(chunk).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=-1)
                    pred = int(torch.argmax(probs, dim=-1).item())
                true_label = source.current_label
                if true_label >= 0:
                    if pred == true_label:
                        correct += 1
                    total += 1
            source.close()
            acc = correct / total if total > 0 else 0.0
            accuracies[subj_name] = (acc, total)
            print(f"  {subj_name}: acc={acc:.4f} ({total} trials)")

        # Aggregate
        acc_vals = [v[0] for v in accuracies.values()]
        print(f"\n{'='*50}")
        print(f"Aggregate: mean={np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}")
        print(f"Best:  {max(acc_vals):.4f} ({max(accuracies, key=lambda k: accuracies[k][0])})")
        print(f"Worst: {min(acc_vals):.4f} ({min(accuracies, key=lambda k: accuracies[k][0])})")
        return

    # ── Source selection ─────────────────────────────────────────
    if args.source == "replay":
        if args.data is None:
            print("ERROR: --data is required with --source replay")
            sys.exit(1)
        from realtime.file_replay import FileReplaySource
        stream = FileReplaySource(
            data_path=args.data,
            labels_path=args.labels,
            trial_mode=True,          # return full trials, bypass ring buffer
        )
        print(f"File replay (trial mode): {args.data}")
    else:
        from realtime.stream import DummyStream
        stream = DummyStream()

    # ── Model loading ────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            sys.exit(1)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        n_classes = cfg.get("n_classes", 3)
        n_times = cfg.get("n_times", 750)
        model_type = (
            "eegnet_spatiotemporal" if "spatiotemporal" in args.checkpoint
            else "eegnet"
        )
        model = create_model(
            model_type,
            n_channels=N_CHANNELS,
            n_classes=n_classes,
            F1=cfg.get("F1", 8),
            D=cfg.get("D", 2),
            F2=cfg.get("F2", 16),
            dropout=cfg.get("dropout", 0.5),
        )
        # Warm-up forward pass to build lazy classifier before load_state_dict
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, N_CHANNELS, n_times))
        model.load_state_dict(checkpoint["state_dict"])
        print(f"  Loaded {model_type} (n_classes={n_classes}) from {args.checkpoint}")
    else:
        n_classes = 3
        model = create_model("eegnet", n_channels=N_CHANNELS, n_classes=n_classes)
    model.eval()

    # ── Inference pipeline ───────────────────────────────────────
    buffer = RingBuffer()
    inference = MIInference(model, buffer, device="cpu", n_classes=n_classes)
    stream.open()
    print("Running inference loop (Ctrl+C to stop)...\n")

    try:
        step = 0
        while True:
            chunk = stream.read_chunk()

            # ── Exhaustion check (must be before inference) ────
            if args.source == "replay" and getattr(stream, "exhausted", False):
                print(f"\nReplay complete ({step} chunks).")
                break

            if args.source == "replay":
                # Trial mode: chunk is a full trial (C, T) — direct inference
                tensor = torch.from_numpy(chunk).unsqueeze(0)  # (1, C, T)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=-1)
                    class_id = int(torch.argmax(probs, dim=-1).item())
                    conf = float(probs.max().item())

                if args.gating:
                    thresh = args.threshold
                    # Auto-detect idle class from action_map
                    idle_cls = next(
                        (k for k, v in inference.action_map.items()
                         if v == "STOP"),
                        None,
                    )
                    if idle_cls is not None and class_id == idle_cls:
                        action = "STOP"
                    elif conf < thresh:
                        action = "STOP"
                    else:
                        action = inference.action_map.get(class_id, "STOP")
                else:
                    action = inference.action_map.get(class_id, "STOP")
            else:
                # Streaming mode: push chunk → buffer → inference
                buffer.push(chunk)
                if args.gating:
                    action, class_id, conf = inference.predict_with_gating(
                        threshold=args.threshold
                    )
                else:
                    class_id, conf = inference.predict()
                    action = inference.action_map.get(class_id, "STOP")

            class_name = inference.action_map.get(class_id, f"CLS_{class_id}")
            bar = "#" * int(conf * 20)
            print(
                f"  [{step:3d}] → [{class_name:6s}] "
                f"confidence={conf:.2f} action={action:6s} | {bar}"
            )

            step += 1

            # Exit conditions (exhaustion for replay is checked above)
            if args.source == "dummy" and step >= 100:
                break
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
