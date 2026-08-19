# TEP Anomaly Detector

A Streamlit app that flags anomalous operating windows in Tennessee Eastman
Process (TEP) sensor data using an LSTM autoencoder trained to reconstruct
normal-operation behavior. This was built as a project for CL653 (AI/ML for
Chemical Engineering), and the repo is deliberately just the inference side
of that project: a trained model, a scaler, a threshold, and the app that
serves them.

## 1. What it does

The Tennessee Eastman Process is a simulated chemical plant (a reactor,
condenser, separator, stripper, compressor) that is a standard benchmark in
the process-monitoring literature. Each timestep of simulation data has 52
variables — 41 continuous process measurements (`xmeas`) and 11 manipulated
variables (`xmv`), things like reactor pressure, feed flow rates, and valve
positions.

The app takes a CSV of these 52 columns, slides a 50-timestep window across
it (stride 1), and asks an autoencoder to reconstruct each window. The
autoencoder was trained only on normal-operation data, so it reconstructs
normal windows well and unfamiliar (faulty) windows poorly. The mean
absolute reconstruction error per window is compared against a fixed
threshold — above it, the window is flagged as anomalous. `app.py` reports
the count and percentage of anomalous windows and plots the error curve
against the threshold line.

## 2. The two models, and why one replaced the other

The git history shows this project went through a real model change, not
just a naming cleanup. The first working version (`189724b`) loaded
`transformer_autoencoder_best.keras` — an encoder-decoder built from
stacked multi-head self-attention blocks (174,196 parameters, paired with its own
empirically chosen threshold of 0.016840).

The very next commit (`f0ea7b2`, "switch to LSTM model") replaced it with
`lstm_autoencoder.h5`, a more conventional stacked-LSTM encoder-decoder:

```
Input (50, 52)
 -> LSTM(128) -> LSTM(64) -> Dense(32)      [encoder, bottleneck = 32]
 -> RepeatVector(50)
 -> LSTM(64) -> LSTM(128) -> TimeDistributed(Dense(52))   [decoder]
```

274,516 parameters, threshold 0.009919. The commit doesn't say why the
switch happened, but the five commits right after it explain the real
reason: getting the Transformer model to deserialize reliably on Streamlit
Cloud's Keras 3 runtime turned out to be a fight (see below), and the LSTM
model — saved in the older, more forgiving HDF5 (`.h5`) format — loaded
without the same fuss. `transformer_autoencoder_best.keras` is still
sitting in the repo; it's dead weight now since `app.py` only ever loads
the LSTM model. I left it in rather than deleting it since removing files
wasn't in scope for this cleanup, but it's worth knowing it's unused.

Each model has its own threshold because reconstruction error scale is
model-specific — you can't reuse one model's threshold on the other's
error distribution.

## 3. Deployment gotchas (from the actual commit log)

Six of the twelve commits in this repo are deployment fixes, not feature
work, and they tell a fairly specific story about shipping a Keras model
to Streamlit Cloud:

- `6dba64e` — **`Load model with compile=False to fix loss deserialization error`.**
  The saved model's loss function couldn't be reconstructed from its config
  on load; skipping compilation (which the app doesn't need anyway, since
  it only calls `.predict()`) sidesteps that entirely.
- `076b2d2` — **`Use tensorflow-cpu 2.17.0 for Keras 3 compatibility`.**
  Pinning the exact TF/Keras version so the model that was saved with one
  Keras version loads correctly on Streamlit Cloud's Keras version — model
  files are not always forward/backward compatible across major Keras
  versions.
- `8a2e878` / `ca8f91d` — **`Add runtime.txt to force Python 3.11`.**
  Streamlit Cloud will pick a default Python version if you don't specify
  one, and that default didn't match what the pinned TensorFlow build
  supported.

I re-ran this myself: with TensorFlow 2.21 installed locally (newer than
the `2.17.0` pinned in `requirements.txt`), both `.h5` and `.keras` files
still loaded cleanly with `compile=False`. So the pins in this repo are
narrower than what's strictly required — they're pinned to whatever
combination the author confirmed worked on Streamlit Cloud at the time,
which is a reasonable and common way to stop fighting version drift once
you've found a working combination, even if it's not the loosest possible
constraint.

## 4. What I verified, and what I couldn't

I don't have the original TEP training/validation data or the training
notebook — they aren't in this repo, only the exported model artifacts
are. So I can't reproduce precision/recall/F1 numbers, and I'm not going
to invent any. What I did verify directly:

- Both `lstm_autoencoder.h5` and `transformer_autoencoder_best.keras` load
  successfully with `tf.keras.models.load_model(..., compile=False)`, and
  their architectures match what's described above.
- `tep_scaler.pkl` unpickles to a fitted `sklearn.preprocessing.MinMaxScaler`
  with `n_features_in_ = 52`.
- `anomaly_threshold.npy` contains `0.009918835743437117` — this matches
  the `THRESHOLD = 0.009919` hardcoded in `app.py` to six decimal places.
  Worth noting: the app doesn't actually load this `.npy` file at runtime;
  the threshold is hardcoded separately and just happens to agree with it.
- I ran the full inference pipeline end-to-end against synthetic data:
  random per-feature noise centered on each feature's real min/max
  midpoint (from the scaler's fitted `data_min_`/`data_max_`, i.e.
  individually plausible sensor values) still produced a mean
  reconstruction error of ~0.041 — over 4x the 0.0099 threshold — and
  flagged 100% of windows as anomalous. That's a useful negative result:
  it means the model isn't just checking that each of the 52 values falls
  in a plausible range independently, it learned the temporal and
  cross-sensor correlation structure of real TEP normal operation, and
  synthetic data that violates that structure gets rejected even when
  every individual value looks reasonable in isolation. I can't confirm
  the model's true detection accuracy without real TEP fault data, but
  this at least confirms it isn't trivially rubber-stamping anything
  in-range.
- I ran `streamlit run app.py` locally and confirmed the server starts and
  responds `HTTP 200` on its port — the app itself boots and serves
  correctly, independent of the modeling question above.

There is no automated test suite in this repo.

## 5. Running it locally

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV with the 52 TEP sensor/actuator columns (any `faultNumber`,
`simulationRun`, or `sample` columns are dropped automatically if present)
and at least 50 rows. `runtime.txt` and `.python-version` pin Python 3.11,
matching what's known to work with the pinned `tensorflow-cpu==2.17.0`
build on Streamlit Cloud.

## 6. Project layout

```
tep-anomaly-detector/
├── app.py                             # Streamlit app: load model + scaler, window, reconstruct, threshold, plot
├── lstm_autoencoder.h5                # active model — stacked LSTM encoder-decoder (274,516 params)
├── transformer_autoencoder_best.keras # earlier model, unused by app.py, kept for reference
├── tep_scaler.pkl                     # fitted sklearn MinMaxScaler, 52 features
├── anomaly_threshold.npy              # threshold value from training; not loaded by app.py (see section 4)
├── requirements.txt                   # streamlit, tensorflow-cpu==2.17.0, numpy, pandas, scikit-learn, matplotlib
├── runtime.txt / .python-version      # pin Python 3.11 for Streamlit Cloud
└── .devcontainer/devcontainer.json    # Codespaces config, auto-runs `streamlit run app.py` on attach
```
