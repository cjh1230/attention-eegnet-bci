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
        "history": np.zeros((N_CHANNELS, 100)),
        "session_log": [],  # [(timestamp, class_id, confidence, label)]
        "confusion_counts": np.zeros((3, 3), dtype=int),
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
        ckpt = selected_ckpt if selected_ckpt != "(untrained)" else None
        st.session_state["model"] = _load_model(ckpt)
        from realtime.buffer import RingBuffer
        st.session_state["buffer"] = RingBuffer()
        st.session_state["session_log"] = []
        st.session_state["confusion_counts"] = np.zeros((3, 3), dtype=int)
        st.success(f"Model loaded{' (' + ckpt + ')' if ckpt else ' (untrained)'}. Ready.")

    # --- Main layout ---
    col1, col2 = st.columns([3, 2])

    if running and st.session_state.get("model") is not None:
        import torch

        model = st.session_state["model"]
        buffer = st.session_state["buffer"]
        history = st.session_state["history"]

        labels = {0: "IDLE", 1: "LEFT", 2: "RIGHT"}
        colors = {0: "gray", 1: "blue", 2: "red"}

        # Synthetic data chunk (replace with real LSL stream when available)
        chunk = np.random.randn(N_CHANNELS, int(250 * 0.125)) * 5.0
        buffer.push(chunk)
        data = buffer.read()

        # Rolling display update
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
                    index=[f"True {labels[i]}" for i in range(3)],
                    columns=[f"Pred {labels[i]}" for i in range(3)],
                )
                st.dataframe(cm_df, use_container_width=True)

        # ---- Column 2: Prediction + Confidence ----
        with col2:
            # Inference
            tensor = torch.from_numpy(data).unsqueeze(0).float()
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=-1).squeeze().numpy()

            class_id = int(np.argmax(probs))
            conf = float(probs[class_id])

            # Update session log
            st.session_state["session_log"].append((
                time.strftime("%H:%M:%S"), class_id, round(conf, 4),
                labels[class_id],
            ))

            # Highlight if above threshold
            above_threshold = conf >= threshold
            emoji = {0: "🟡", 1: "👈", 2: "👉"}
            st.subheader("Prediction")
            st.markdown(
                f"## {emoji.get(class_id, '❓')} {labels.get(class_id, '?')}"
            )
            if above_threshold:
                st.success(f"Confidence: {conf:.2%} ✅")
            else:
                st.warning(f"Confidence: {conf:.2%} ⚠️ (below threshold)")

            st.subheader("Class Probabilities")
            prob_data = {labels[i]: float(probs[i]) for i in range(len(labels))}
            st.bar_chart(prob_data)

            st.subheader("Recent Log")
            log = st.session_state["session_log"]
            for entry in log[-10:]:
                ts, cid, cf, lbl = entry
                st.text(f"[{ts}] {lbl:5s} | conf={cf:.3f}")

        # Re-run after a short delay
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
