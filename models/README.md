# ML Models (F3 Face rPPG + Finger PPG) — Scaffolding

These two files (`face_rppg.py`, `finger_ppg.py`) are **PyTorch architecture-only**
reference implementations for future server-side inference:

| File | Model | Outputs | Inputs |
|------|-------|---------|--------|
| `face_rppg.py` | Face rPPG — EfficientNet-B0 + TCN + Temporal Attention | BVP + Respiratory waveforms → HR, HRV, RR, quality | (B, T, 3, 72, 72) face crops |
| `finger_ppg.py` | Finger PPG — 1D CNN + Bi-GRU multi-head | BVP + HR + SpO2 + trend BP | (B, 3, T) RGB channel means |

## Status — NOT YET SERVED

- **No trained weights are checked in.** These files ship only network architectures.
- Running the models on random inputs will produce noise. They MUST be trained on a
  validated rPPG/PPG dataset (e.g., PURE, UBFC-rPPG, MMSE-HR, BP4D+) before serving.
- `torch` / `timm` / `einops` are **not installed** in this backend image. Adding them
  bloats the image ~500-700 MB and is only worthwhile once trained weights exist.

## What we DO serve today

The current DilSay pipeline uses the in-browser vanilla JS `RPPGEstimator`
(`/app/frontend/public/aisteth/rppg.js`) which runs 100 % on the client and posts
fused vitals to `POST /api/snapshot`. This keeps the backend light and privacy-preserving.

A small subset of these Python files is reused today: the **scipy-only** post-processing
helpers (`bvp_to_vitals`, `bvp_to_hrv`) have been extracted into
`/app/backend/signal_utils.py` and are available for future server-side verification of
client-submitted BVP traces — without pulling in PyTorch.

## How to activate this path in the future

1. Train each model on a validated dataset. Save weights to `models/checkpoints/`.
2. Install runtime deps:
   ```bash
   pip install torch timm einops einops-exts scipy
   pip freeze > /app/backend/requirements.txt
   ```
3. Add an inference wrapper (`models/inference.py`) that loads weights lazily on first call.
4. Expose new endpoints:
   - `POST /api/vitals/face-rppg`  → accepts a URL/base64 clip of face frames.
   - `POST /api/vitals/finger-ppg` → accepts a channel-mean RGB trace from the finger.
5. Gate both endpoints behind a feature flag (`ENABLE_ML_INFERENCE=1`) so the browser
   pipeline remains the default.

## Honest disclaimer

Even with trained weights, camera-based vitals are **screening estimates, not medical
measurements**. The BP head in `finger_ppg.py` is a *trend proxy*, not a diagnostic
reading. Any user-facing surface must retain the current wellness/screening disclaimer.
