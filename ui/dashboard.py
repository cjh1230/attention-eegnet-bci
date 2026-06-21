"""
Streamlit dashboard for real-time MI-EEG visualization and inference.

Usage:
    streamlit run ui/dashboard.py
"""
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

from utils.config import N_CHANNELS, CHANNEL_NAMES, BUFFER_WINDOW


def main():
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="MI-BCI Dashboard", layout="wide")
    st.title("🧠 运动想象 BCI 实时监控")
    st.caption("Motor Imagery — Brain-Computer Interface Dashboard")

    # Sidebar controls
    st.sidebar.header("控制面板")
    running = st.sidebar.toggle("启动实时推理", value=False)
    threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.6, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 设备状态")
    st.sidebar.metric("采样率", "250 Hz", help="DeepBCI 默认采样率")
    st.sidebar.metric("通道数", "16")
    st.sidebar.metric("缓冲区窗口", f"{BUFFER_WINDOW}s")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("EEG 波形")
        chart_placeholder = st.empty()

    with col2:
        st.subheader("预测结果")
        pred_placeholder = st.empty()
        st.subheader("置信度")
        conf_placeholder = st.empty()
        st.subheader("动作")
        action_placeholder = st.empty()

    # Simulated data loop
    if running:
        from realtime.buffer import RingBuffer
        from models.eegnet import EEGNet

        buffer = RingBuffer()
        model = EEGNet(n_channels=N_CHANNELS, n_classes=3)
        model.eval()

        history = np.zeros((N_CHANNELS, 100))  # rolling display
        labels = {0: "⏸ Idle", 1: "👈 Left", 2: "👉 Right"}
        colors = {0: "gray", 1: "blue", 2: "red"}

        while running:
            # Synthetic data
            chunk = np.random.randn(N_CHANNELS, int(250 * 0.125)) * 5.0
            buffer.push(chunk)
            data = buffer.read()

            # Rolling display update
            history = np.roll(history, -1, axis=-1)
            history[:, -1] = data[:, 0]

            chart_placeholder.line_chart(history.T, height=300)

            # Inference
            import torch
            tensor = torch.from_numpy(data).unsqueeze(0).float()
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=-1).squeeze().numpy()

            class_id = int(np.argmax(probs))
            conf = float(probs[class_id])

            pred_placeholder.markdown(
                f"### {labels.get(class_id, str(class_id))}"
            )
            conf_placeholder.progress(float(conf))
            action_placeholder.markdown(
                f"# {'←' if class_id == 1 else '→' if class_id == 2 else '−'}"
            )

            time.sleep(0.125)

    else:
        st.info("点击侧边栏「启动实时推理」开始")


if __name__ == "__main__":
    if HAS_STREAMLIT:
        main()
    else:
        import os
        os.system("streamlit run ui/dashboard.py")
