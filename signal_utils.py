"""Server-side signal-processing helpers extracted from the model files.

These are the *pure* scipy/numpy post-processing routines from `models/face_rppg.py`
and `models/finger_ppg.py`. They don't depend on PyTorch and are safe to use in the
live FastAPI server today for validating client-submitted BVP traces.

All functions accept a 1-D numpy array of BVP samples and a sample rate (fps),
and return a dict of derived vitals.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def bvp_to_vitals(bvp: np.ndarray, fps: float = 30.0) -> dict:
    """Extract HR, HRV (RMSSD), RR, and quality proxy from a BVP waveform.

    Matches `models/face_rppg.py::bvp_to_vitals` exactly — keep in sync.
    """
    b, a = sp_signal.butter(3, [0.7, 3.5], btype='bandpass', fs=fps)
    bvp_f = sp_signal.filtfilt(b, a, bvp)

    freqs, psd = sp_signal.welch(bvp_f, fs=fps, nperseg=min(len(bvp_f), 256))
    hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
    peak_hz = freqs[hr_mask][np.argmax(psd[hr_mask])] if hr_mask.any() else 1.2
    hr_bpm = peak_hz * 60.0

    min_distance = int(fps * 0.4)
    peaks, _ = sp_signal.find_peaks(bvp_f, distance=min_distance, prominence=0.01)
    hrv_rmssd = None
    if len(peaks) >= 3:
        ibi_ms = np.diff(peaks) / fps * 1000
        ibi_ms = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 2000)]
        if len(ibi_ms) >= 2:
            hrv_rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))

    b2, a2 = sp_signal.butter(2, [0.15, 0.5], btype='bandpass', fs=fps)
    resp_f = sp_signal.filtfilt(b2, a2, bvp)
    freqs2, psd2 = sp_signal.welch(resp_f, fs=fps, nperseg=min(len(resp_f), 256))
    rr_mask = (freqs2 >= 0.15) & (freqs2 <= 0.5)
    rr_hz = freqs2[rr_mask][np.argmax(psd2[rr_mask])] if rr_mask.any() else 0.25
    rr_bpm = rr_hz * 60.0

    signal_power = psd[hr_mask].max() if hr_mask.any() else 0.0
    noise_power = psd[~hr_mask].mean() + 1e-10
    snr_db = 10 * np.log10(max(signal_power, 1e-10) / noise_power)
    quality = float(np.clip((snr_db - 2) / 18, 0, 1))

    return {
        "hr_bpm": float(hr_bpm),
        "hrv_rmssd_ms": hrv_rmssd,
        "rr_bpm": float(rr_bpm),
        "quality": quality,
    }


def bvp_to_hrv(bvp: np.ndarray, fps: float = 30.0) -> dict:
    """Extract HRV metrics (RMSSD, SDNN, NN50, pNN50) from a PPG BVP waveform.

    Matches `models/finger_ppg.py::bvp_to_hrv` exactly — keep in sync.
    """
    b, a = sp_signal.butter(3, [0.5, 4.0], btype='bandpass', fs=fps)
    bvp_f = sp_signal.filtfilt(b, a, bvp)

    min_dist = int(fps * 0.33)
    peaks, _ = sp_signal.find_peaks(bvp_f, distance=min_dist, prominence=0.05)

    if len(peaks) < 3:
        return {"hrv_rmssd_ms": None, "hrv_sdnn_ms": None,
                "hr_bpm": None, "nn50": None, "pnn50": None}

    ibi_ms = np.diff(peaks) / fps * 1000
    ibi_ms = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 2000)]
    if len(ibi_ms) < 2:
        return {"hrv_rmssd_ms": None, "hrv_sdnn_ms": None,
                "hr_bpm": None, "nn50": None, "pnn50": None}

    rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    sdnn = float(np.std(ibi_ms))
    hr_bpm = float(60000.0 / ibi_ms.mean())
    nn50 = int(np.sum(np.abs(np.diff(ibi_ms)) > 50))
    pnn50 = float(nn50 / max(1, len(ibi_ms) - 1))

    return {
        "hrv_rmssd_ms": rmssd, "hrv_sdnn_ms": sdnn,
        "hr_bpm": hr_bpm, "nn50": nn50, "pnn50": pnn50,
    }
