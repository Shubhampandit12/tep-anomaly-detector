import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TEP Anomaly Detector", page_icon="⚗️", layout="wide")
st.title("⚗️ Tennessee Eastman Process — Anomaly Detector")
st.markdown("Upload a sequence of process sensor readings to check for anomalies.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('lstm_autoencoder.h5', compile=False)
    return model

@st.cache_resource
def load_scaler():
    with open('tep_scaler.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()
THRESHOLD = 0.009919

with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("**Model:** LSTM Autoencoder")
    st.write("**Dataset:** Tennessee Eastman Process")
    st.write("**Input Shape:** (50 timesteps × 52 features)")
    st.write(f"**Anomaly Threshold:** {THRESHOLD:.6f}")
    st.markdown("---")
    st.write("**Course:** CL653 — AI/ML for Chemical Engineering")
    st.write("**Student:** Shubham Pandit")

st.subheader("📂 Upload your CSV file")
uploaded_file = st.file_uploader("Upload a CSV with 52 sensor columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    drop_cols = [c for c in ['faultNumber', 'simulationRun', 'sample'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    if df.shape[1] != 52:
        st.error(f"Expected 52 feature columns, but got {df.shape[1]}. Please check your file.")
    elif len(df) < 50:
        st.error("Need at least 50 rows to form one window.")
    else:
        data = scaler.transform(df.values)
        WINDOW = 50
        sequences = []
        for i in range(0, len(data) - WINDOW + 1):
            sequences.append(data[i:i+WINDOW])
        sequences = np.array(sequences)

        reconstructed = model.predict(sequences, verbose=0)
        errors = np.mean(np.mean(np.abs(sequences - reconstructed), axis=2), axis=1)
        anomalies = errors > THRESHOLD
        n_anomalies = int(np.sum(anomalies))
        anomaly_pct = 100 * n_anomalies / len(errors)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Windows", len(errors))
        col2.metric("Anomalous Windows", n_anomalies)
        col3.metric("Anomaly %", f"{anomaly_pct:.1f}%")

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(errors, label="Reconstruction Error", color="steelblue", linewidth=0.8)
        ax.axhline(THRESHOLD, color="red", linestyle="--", label=f"Threshold ({THRESHOLD:.4f})")
        ax.fill_between(range(len(errors)), errors, THRESHOLD,
                        where=(errors > THRESHOLD), alpha=0.3, color="red", label="Anomaly")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("MAE")
        ax.set_title("Reconstruction Error over Time")
        ax.legend()
        st.pyplot(fig)

        if n_anomalies == 0:
            st.success("✅ No anomalies detected — process looks normal.")
        else:
            st.warning(f"⚠️ {n_anomalies} anomalous window(s) detected!")
else:
    st.info("👆 Upload a CSV file to begin analysis.")
