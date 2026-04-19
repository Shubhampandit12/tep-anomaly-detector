import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TEP Anomaly Detector",
    page_icon="⚗️",
    layout="wide"
)

st.title("⚗️ Tennessee Eastman Process — Anomaly Detector")
st.markdown("Upload a sequence of process sensor readings to check for anomalies.")

# ── Load Model & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
   model = tf.keras.models.load_model('lstm_autoencoder.h5')
    return model

@st.cache_resource
def load_scaler():
    with open('tep_scaler.pkl', 'rb') as f:
        return pickle.load(f)

model     = load_model()
scaler    = load_scaler()
THRESHOLD = 0.009919
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("**Model:** LSTM Autoencoder")
    st.write("**Dataset:** Tennessee Eastman Process")
    st.write("**Input Shape:** (50 timesteps × 52 features)")
    st.write(f"**Anomaly Threshold:** {THRESHOLD:.6f}")
    st.markdown("---")
    st.write("**Course:** CL653 — AI/ML for Chemical Engineering")
    st.write("**Student:** Shubham Pandit")

# ── File Upload ───────────────────────────────────────────────────────────────
st.subheader("📂 Upload Process Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file with 52 process features (at least 50 rows)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    feature_cols = [c for c in df.columns if c.startswith('xmeas') or c.startswith('xmv')]
    df = df[feature_cols]

    st.write(f"Loaded: **{df.shape[0]} rows × {df.shape[1]} features**")

    if df.shape[1] != 52:
        st.error("❌ Expected 52 feature columns (xmeas_1–41 + xmv_1–11). Check your file.")
    elif df.shape[0] < 50:
        st.error("❌ Need at least 50 rows to form one window.")
    else:
        scaled = scaler.transform(df.values)

        windows = []
        for i in range(len(scaled) - 49):
            windows.append(scaled[i:i+50])
        X = np.array(windows)

        st.write(f"Created **{len(X)} windows** of shape (50, 52)")

        with st.spinner("Running anomaly detection..."):
            X_pred = model.predict(X, batch_size=64, verbose=0)
            errors = np.mean((X - X_pred) ** 2, axis=(1, 2))

        n_anomalies = np.sum(errors > THRESHOLD)
        pct         = n_anomalies / len(errors) * 100

        st.markdown("---")
        st.subheader("🔍 Detection Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows",     len(errors))
        col2.metric("Anomalies Flagged", n_anomalies)
        col3.metric("Anomaly Rate",      f"{pct:.2f}%")

        if pct > 5:
            st.error("🚨 HIGH ANOMALY RATE — Process fault likely present!")
        elif pct > 1:
            st.warning("⚠️ MODERATE ANOMALY RATE — Investigate process conditions.")
        else:
            st.success("✅ NORMAL OPERATION — Reconstruction error within threshold.")

        st.subheader("📈 Reconstruction Error Timeline")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(errors, color='steelblue', linewidth=0.8, label='MSE per Window')
        ax.axhline(THRESHOLD, color='red', linestyle='--',
                   linewidth=1.5, label=f'Threshold ({THRESHOLD:.4f})')
        ax.fill_between(range(len(errors)), THRESHOLD, errors,
                        where=(errors > THRESHOLD),
                        color='red', alpha=0.3, label='Anomaly Detected')
        ax.set_xlabel('Window Index')
        ax.set_ylabel('MSE')
        ax.set_title('Reconstruction Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to begin. Use any subset of the TEP dataset.")