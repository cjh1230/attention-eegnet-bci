"""
Streamlit dashboard for real-time MI-EEG visualization and inference.

Usage:
    streamlit run ui/dashboard.py
"""
import csv
import io
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from utils.config import N_CHANNELS, BUFFER_WINDOW


def _get_available_checkpoints():
    """List all .pt checkpoint files in checkpoints/."""
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted([p.name for p in ckpt_dir.glob("*.pt")])


def _load_model(checkpoint_name: str = None):
    """Load a trained model checkpoint, or return an untrained model."""
    from models.eegnet import EEGNet
    from models.eegnet_attn import create_model

    if checkpoint_name:
        ckpt_path = ROOT / "checkpoints" / checkpoint_name
        # Try generic load
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        n_channels = cfg.get("n_channels", N_CHANNELS)
        n_classes = cfg.get("n_classes", 3)

        # Detect if this is an attention model by checking checkpoint name
        if "eegnet_best" in checkpoint_name or checkpoint_name == "eegnet_best.pt":
            model = EEGNet(
                n_channels=n_channels, n_classes=n_classes,
                F1=cfg.get("F1", 8), D=cfg.get("D", 2), F2=cfg.get("F2", 16),
                dropout=cfg.get("dropout", 0.5),
            )
        else:
            # Try to infer model type from filename
            for mt in ["eegnet_spatiotemporal", "eegnet_mhsa", "eegnet_se",
                        "eegnet_temporal"]:
                if mt in checkpoint_name:
                    model = create_model(mt, n_channels=n_channels, n_classes=n_classes)
                    break
            else:
                model = EEGNet(n_channels=n_channels, n_classes=n_classes)

        # Warm-up forward for lazy classifier
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, n_channels, cfg.get("n_times", 750)))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    else:
        st.warning("No trained model found. Predictions are random. Run `python main.py train` first.")
        from models.eegnet import EEGNet
        model = EEGNet(n_channels=N_CHANNELS, n_classes=3)
        model.eval()
        return model


def _export_session_csv(history):
    """Export session history as CSV bytes."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "class_id", "confidence", "class_label"])
    for entry in history:
        writer.writerow(entry)
    return output.getvalue().encode("utf-8")


def main():
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="MI-BCI Dashboard", layout="wide")
    st.title("🧠 MI-BCI Real-time Monitor")
    st.caption("Motor Imagery — Brain-Computer Interface Dashboard")

    # --- Session state init ---
    defaults = {
        "buffer": None,
        "model": None,
        "n_classes": 3,
        "history": np.zeros((N_CHANNELS, 100)),
        "session_log": [],
        "confusion_counts": np.zeros((3, 3), dtype=int),
        "replay_source": None,
        "replay_trial_idx": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # --- Sidebar ---
    st.sidebar.header("Controls")

    # Model selection
    checkpoints = _get_available_checkpoints()
    checkpoint_options = ["(untrained)"] + checkpoints
    selected_ckpt = st.sidebar.selectbox("Model Checkpoint", checkpoint_options)

    # Data source
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Source Type",
        ["Synthetic (random)", "File Replay"],
        index=0,
    )
    replay_path = ""
    if data_source == "File Replay":
        replay_path = st.sidebar.text_input(
            "Path to X.npy",
            value="data/loso_binary/subj_01/X.npy",
        )

    startup = st.sidebar.button("Load Model & Start", type="primary")
    stop = st.sidebar.button("Stop")

    running = st.sidebar.toggle("Streaming", value=False)

    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)

    # Session export
    if st.session_state["session_log"]:
        csv_data = _export_session_csv(st.session_state["session_log"])
        st.sidebar.download_button(
            label="📥 Export Session (CSV)",
            data=csv_data,
            file_name=f"bci_session_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Device Status")
    st.sidebar.metric("Sample Rate", "250 Hz", help="DeepBCI default")
    st.sidebar.metric("Channels", f"{N_CHANNELS}")
    st.sidebar.metric("Buffer Window", f"{BUFFER_WINDOW}s")

    if stop:
        st.session_state["buffer"] = None
        st.session_state["model"] = None
        st.info("Stopped. Click 'Load Model & Start' to begin.")

    if startup:
        ckpt_name = selected_ckpt if selected_ckpt != "(untrained)" else None
        model = _load_model(ckpt_name)

        # Determine n_classes from model
        import torch
        if hasattr(model, "n_classes"):
            n_classes = model.n_classes
        elif hasattr(model, "classifier") and model.classifier is not None:
            n_classes = model.classifier[-1].out_features
        else:
            n_classes = 3

        # Build dynamic action map
        from realtime.inference import build_action_map
        action_map = build_action_map(n_classes, "physionet_mi")
        labels = dict(action_map)
        colors = {}
        for i, name in labels.items():
            if name == "STOP":
                colors[i] = "gray"
            elif "LEFT" in name:
                colors[i] = "blue"
            elif "RIGHT" in name:
                colors[i] = "red"
            else:
                colors[i] = "green"

        st.session_state["model"] = model
        st.session_state["n_classes"] = n_classes
        st.session_state["labels"] = labels
        st.session_state["colors"] = colors
        st.session_state["action_map"] = action_map

        from realtime.buffer import RingBuffer
        st.session_state["buffer"] = RingBuffer()
        st.session_state["session_log"] = []
        st.session_state["confusion_counts"] = np.zeros(
            (n_classes, n_classes), dtype=int
        )

        # Initialize replay source if selected
        if data_source == "File Replay" and replay_path:
            from realtime.file_replay import FileReplaySource
            source = FileReplaySource(data_path=replay_path, trial_mode=True)
            source.open()
            st.session_state["replay_source"] = source
            st.session_state["replay_trial_idx"] = 0
        else:
            st.session_state["replay_source"] = None

        ckpt_label = f" ({ckpt_name})" if ckpt_name else " (untrained)"
        st.success(f"Model loaded{ckpt_label} — {n_classes} classes. Ready.")

    # --- Main layout ---
    col1, col2 = st.columns([3, 2])

    if running and st.session_state.get("model") is not None:
        import torch

        model = st.session_state["model"]
        buffer = st.session_state["buffer"]
        history = st.session_state["history"]
        replay_src = st.session_state.get("replay_source")

        labels = st.session_state.get("labels", {0: "STOP", 1: "LEFT", 2: "RIGHT"})
        colors = st.session_state.get("colors", {0: "gray", 1: "blue", 2: "red"})
        n_classes = st.session_state.get("n_classes", 3)

        # --- Data acquisition ---
        true_label = None
        if replay_src is not None:
            # File replay: feed full trials directly
            chunk = replay_src.read_chunk()
            if replay_src.exhausted:
                st.warning("Replay exhausted. Toggle Streaming off/on to restart.")
                chunk = np.random.randn(N_CHANNELS, int(250 * 0.125)) * 5.0
            else:
                true_label = replay_src.current_label
                # Bypass ring buffer: feed full trial to model
                tensor = torch.from_numpy(chunk).unsqueeze(0).float()
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=-1).squeeze().numpy()
                class_id = int(np.argmax(probs))
                conf = float(probs[class_id])

                # Update session log
                st.session_state["session_log"].append((
                    time.strftime("%H:%M:%S"), class_id, round(conf, 4),
                    labels.get(class_id, "?"),
                ))

                # Update confusion matrix with true label if available
                if true_label is not None and true_label >= 0:
                    cm = st.session_state["confusion_counts"]
                    if true_label < n_classes and class_id < n_classes:
                        cm[true_label, class_id] += 1

                # Waveform: show the full trial (first channel, subsampled)
                history = np.roll(history, -1, axis=-1)
                history[:, -1] = chunk[:, 0]  # first channel, first sample
                st.session_state["history"] = history

                # Show results
                with col1:
                    st.subheader("EEG Waveform (replay)")
                    st.line_chart(history.T, height=250)

                    st.subheader("Confusion Matrix (cumulative)")
                    cm = st.session_state["confusion_counts"]
                    if cm.sum() > 0:
                        import pandas as pd
                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"True {labels[i]}" for i in range(n_classes)],
                            columns=[f"Pred {labels[i]}" for i in range(n_classes)],
                        )
                        st.dataframe(cm_df, use_container_width=True)

                with col2:
                    above_threshold = conf >= threshold
                    emoji_map = {0: "🟡", 1: "👈", 2: "👉", 3: "🦶", 4: "👅"}
                    st.subheader("Prediction")
                    st.markdown(
                        f"## {emoji_map.get(class_id, '❓')} {labels.get(class_id, '?')}"
                    )
                    if true_label is not None and true_label >= 0:
                        true_name = labels.get(true_label, f"CLS_{true_label}")
                        st.caption(f"True label: {true_name}")
                    if above_threshold:
                        st.success(f"Confidence: {conf:.2%} ✅")
                    else:
                        st.warning(f"Confidence: {conf:.2%} ⚠️ (below threshold)")

                    st.subheader("Class Probabilities")
                    prob_data = {labels.get(i, f"CLS_{i}"): float(probs[i])
                                 for i in range(len(probs))}
                    st.bar_chart(prob_data)

                    st.subheader("Recent Log")
                    log = st.session_state["session_log"]
                    for entry in log[-10:]:
                        ts, cid, cf, lbl = entry
                        st.text(f"[{ts}] {lbl:5s} | conf={cf:.3f}")

                time.sleep(0.125)
                st.rerun()

        else:
            # Synthetic data: ring buffer → inference
            chunk = np.random.randn(N_CHANNELS, int(250 * 0.125)) * 5.0
            buffer.push(chunk)
            data = buffer.read()

            history = np.roll(history, -1, axis=-1)
            history[:, -1] = data[:, 0]
            st.session_state["history"] = history

            # ---- Column 1: Waveform + Confusion Matrix ----
            with col1:
                st.subheader("EEG Waveform (live)")
                st.line_chart(history.T, height=250)

                st.subheader("Confusion Matrix (cumulative)")
                cm = st.session_state["confusion_counts"]
                if cm.sum() > 0:
                    import pandas as pd
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"True {labels[i]}" for i in range(n_classes)],
                        columns=[f"Pred {labels[i]}" for i in range(n_classes)],
                    )
                    st.dataframe(cm_df, use_container_width=True)

            # ---- Column 2: Prediction + Confidence ----
            with col2:
                tensor = torch.from_numpy(data).unsqueeze(0).float()
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=-1).squeeze().numpy()

                class_id = int(np.argmax(probs))
                conf = float(probs[class_id])

                st.session_state["session_log"].append((
                    time.strftime("%H:%M:%S"), class_id, round(conf, 4),
                    labels.get(class_id, "?"),
                ))

                above_threshold = conf >= threshold
                emoji_map = {0: "🟡", 1: "👈", 2: "👉", 3: "🦶", 4: "👅"}
                st.subheader("Prediction")
                st.markdown(
                    f"## {emoji_map.get(class_id, '❓')} {labels.get(class_id, '?')}"
                )
                if above_threshold:
                    st.success(f"Confidence: {conf:.2%} ✅")
                else:
                    st.warning(f"Confidence: {conf:.2%} ⚠️ (below threshold)")

                st.subheader("Class Probabilities")
                prob_data = {labels.get(i, f"CLS_{i}"): float(probs[i])
                             for i in range(len(probs))}
                st.bar_chart(prob_data)

                st.subheader("Recent Log")
                log = st.session_state["session_log"]
                for entry in log[-10:]:
                    ts, cid, cf, lbl = entry
                    st.text(f"[{ts}] {lbl:5s} | conf={cf:.3f}")

            time.sleep(0.125)
            st.rerun()

    elif not running:
        # Show model info when idle
        if st.session_state.get("model") is not None:
            with col1:
                st.info("Toggle 'Streaming' in the sidebar to start inference.")
                st.subheader("EEG Waveform (last session)")
                st.line_chart(st.session_state["history"].T, height=250)
        else:
            st.info("Click 'Load Model & Start' in the sidebar to begin.")


if __name__ == "__main__":
    if not HAS_STREAMLIT:
        raise ImportError("Streamlit not installed. Run: pip install streamlit")
    main()
