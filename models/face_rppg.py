"""
F3 Face rPPG Model
==================
Estimates HR, HRV (RMSSD), and RR (respiratory rate) from facial video.

Architecture: Temporal Difference + EfficientNet-B0 + Temporal Convolutional Network
- Input : (B, T, 3, H, W) — T consecutive RGB frames, face-cropped, 72×72
- Output: (B, T-1) BVP signal + (B, 1) respiratory signal

Post-processing (inference only, not trainable):
  HR    = peak frequency of BVP via Welch PSD  [bpm]
  HRV   = RMSSD of inter-beat intervals         [ms]
  RR    = peak frequency of resp signal          [breaths/min]

References:
  EfficientPhys: Liu et al., CVPR Workshop 2023
  PhysNet: Yu et al., ICCV 2019
  TSCAN: Liu et al., NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
import numpy as np
from scipy import signal as sp_signal


# ── Temporal Difference Preprocessing ────────────────────────────────────────

class TemporalDifference(nn.Module):
    """
    Replace raw frames with normalised temporal differences.
    Removes illumination bias and amplifies pulse-driven colour change.
    d[t] = (f[t+1] - f[t]) / (f[t+1] + f[t] + 1e-7)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        d = (x[:, 1:] - x[:, :-1]) / (x[:, 1:] + x[:, :-1] + 1e-7)
        return d  # (B, T-1, C, H, W)


# ── Squeeze-and-Excitation Temporal Attention ─────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Channel + temporal attention over the feature sequence.
    Lets the model down-weight frames with motion artifacts.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        b, c, h, w = x.shape
        w_vec = self.gap(x).view(b, c)
        w_vec = self.fc(w_vec).view(b, c, 1, 1)
        return x * w_vec


# ── Temporal Convolutional Block ──────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Dilated causal 1-D convolution for sequence modelling."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)[:, :, :x.size(2)]  # causal trim
        return self.act(self.norm(out)) + self.skip(x)


# ── Main Model ────────────────────────────────────────────────────────────────

class FaceRPPGModel(nn.Module):
    """
    Multi-task rPPG model.

    Outputs
    -------
    bvp  : (B, T-1) — normalised BVP waveform
    resp : (B, T-1) — respiratory waveform (derived from RSA)
    """

    def __init__(
        self,
        frames: int = 160,       # number of input frames
        img_size: int = 72,      # spatial crop size
        feat_dim: int = 256,     # TCN hidden dim
        pretrained: bool = True,
    ):
        super().__init__()
        self.frames   = frames
        self.img_size = img_size

        # Preprocessing
        self.td = TemporalDifference()

        # Spatial feature extractor — EfficientNet-B0, strip classifier
        eff = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, features_only=True
        )
        # Use only first 3 stages to stay lightweight
        self.spatial = nn.Sequential(*list(eff.children())[:3])
        spatial_out  = 40  # EfficientNet-B0 stage-2 channels

        # Spatial pooling + attention
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.sa     = TemporalAttention(spatial_out)
        self.proj   = nn.Conv1d(spatial_out, feat_dim, 1)

        # Temporal convolutional network (multi-scale dilations)
        self.tcn = nn.Sequential(
            TCNBlock(feat_dim, feat_dim, kernel=3, dilation=1),
            TCNBlock(feat_dim, feat_dim, kernel=3, dilation=2),
            TCNBlock(feat_dim, feat_dim, kernel=3, dilation=4),
            TCNBlock(feat_dim, feat_dim, kernel=3, dilation=8),
        )

        # Output heads
        self.bvp_head  = nn.Conv1d(feat_dim, 1, 1)   # BVP signal
        self.resp_head = nn.Conv1d(feat_dim, 1, 1)   # Respiratory signal

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, 3, H, W)   float32, pixels in [0, 1]
        """
        B, T, C, H, W = x.shape

        # Temporal difference: (B, T-1, C, H, W)
        d = self.td(x)
        T1 = T - 1

        # Spatial features per frame — batch over time
        d_flat = rearrange(d, 'b t c h w -> (b t) c h w')
        feat   = self.spatial(d_flat)          # (B*T1, C', H', W')
        feat   = self.sa(feat)
        feat   = self.pool(feat).squeeze(-1).squeeze(-1)  # (B*T1, C')
        feat   = rearrange(feat, '(b t) c -> b c t', b=B, t=T1)  # (B, C', T1)

        # Project to TCN dim
        feat = self.proj(feat)                 # (B, feat_dim, T1)

        # Temporal modelling
        feat = self.tcn(feat)                  # (B, feat_dim, T1)

        # Heads
        bvp  = self.bvp_head(feat).squeeze(1)   # (B, T1)
        resp = self.resp_head(feat).squeeze(1)  # (B, T1)

        return bvp, resp


# ── Loss Functions ────────────────────────────────────────────────────────────

def negative_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Pearson correlation — standard loss for rPPG.
    Robust to amplitude differences; only cares about waveform shape.
    pred, target: (B, T)
    """
    pred_m   = pred   - pred.mean(dim=1, keepdim=True)
    target_m = target - target.mean(dim=1, keepdim=True)
    num  = (pred_m * target_m).sum(dim=1)
    denom = torch.sqrt((pred_m**2).sum(dim=1) * (target_m**2).sum(dim=1) + 1e-8)
    r = num / denom
    return (-r).mean()


def hr_mae_loss(pred_bvp: torch.Tensor, target_hr: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """
    HR auxiliary loss: estimate HR from predicted BVP via FFT,
    compare against ground-truth HR label.
    pred_bvp : (B, T)
    target_hr: (B,)  bpm
    """
    T = pred_bvp.shape[1]
    fft = torch.fft.rfft(pred_bvp, dim=1)
    power = torch.abs(fft) ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0/fps)   # Hz
    # HR band: 0.7–3.5 Hz (42–210 bpm)
    mask  = (freqs >= 0.7) & (freqs <= 3.5)
    power_masked = power * mask.unsqueeze(0)
    peak_freq = freqs[power_masked.argmax(dim=1)]  # Hz
    pred_hr   = peak_freq * 60.0                   # bpm
    return F.l1_loss(pred_hr, target_hr)


def skin_tone_equity_loss(losses_per_sample: torch.Tensor, fitzpatrick: torch.Tensor) -> torch.Tensor:
    """
    Penalise variance in per-sample loss across Fitzpatrick skin tone groups.
    Encourages the model to perform equally well across I–VI.

    losses_per_sample : (B,) individual sample losses
    fitzpatrick       : (B,) int tensor, values 1–6
    """
    group_means = []
    for fitz in range(1, 7):
        mask = fitzpatrick == fitz
        if mask.sum() > 0:
            group_means.append(losses_per_sample[mask].mean())
    if len(group_means) < 2:
        return torch.tensor(0.0, device=losses_per_sample.device)
    group_means = torch.stack(group_means)
    return group_means.var()


class RPPGLoss(nn.Module):
    """Combined loss for face rPPG training."""
    def __init__(
        self,
        w_pearson: float = 1.0,
        w_hr:      float = 0.5,
        w_resp:    float = 0.3,
        w_equity:  float = 0.2,
    ):
        super().__init__()
        self.w_pearson = w_pearson
        self.w_hr      = w_hr
        self.w_resp    = w_resp
        self.w_equity  = w_equity

    def forward(
        self,
        pred_bvp:   torch.Tensor,   # (B, T)
        pred_resp:  torch.Tensor,   # (B, T)
        gt_bvp:     torch.Tensor,   # (B, T)
        gt_hr:      torch.Tensor,   # (B,) bpm
        gt_resp:    torch.Tensor,   # (B,) breaths/min
        fitzpatrick: torch.Tensor,  # (B,) int 1-6
        fps: float = 30.0,
    ) -> dict:

        # Per-sample Pearson loss for equity tracking
        B = pred_bvp.shape[0]
        pearson_per = torch.stack([
            negative_pearson(pred_bvp[i:i+1], gt_bvp[i:i+1])
            for i in range(B)
        ])

        l_pearson = pearson_per.mean()
        l_hr      = hr_mae_loss(pred_bvp, gt_hr, fps)
        l_resp    = hr_mae_loss(pred_resp * 60, gt_resp, fps)  # rescale resp
        l_equity  = skin_tone_equity_loss(pearson_per.abs(), fitzpatrick)

        total = (
            self.w_pearson * l_pearson
            + self.w_hr    * l_hr
            + self.w_resp  * l_resp
            + self.w_equity * l_equity
        )

        return {
            "loss":     total,
            "pearson":  l_pearson.item(),
            "hr_mae":   l_hr.item(),
            "resp_mae": l_resp.item(),
            "equity":   l_equity.item(),
        }


# ── Post-processing: BVP → HR, HRV, RR ───────────────────────────────────────

def bvp_to_vitals(bvp: np.ndarray, fps: float = 30.0) -> dict:
    """
    Extract HR, HRV (RMSSD), and RR from a BVP waveform.
    Uses scipy signal processing — no ML.

    bvp : (T,) array, any scale
    fps : frames per second of the original video

    Returns dict with hr_bpm, hrv_rmssd_ms, rr_bpm, quality
    """
    # --- Bandpass filter BVP (0.7–3.5 Hz) ---
    b, a = sp_signal.butter(3, [0.7, 3.5], btype='bandpass', fs=fps)
    bvp_f = sp_signal.filtfilt(b, a, bvp)

    # --- HR via Welch PSD ---
    freqs, psd = sp_signal.welch(bvp_f, fs=fps, nperseg=min(len(bvp_f), 256))
    hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
    peak_hz = freqs[hr_mask][np.argmax(psd[hr_mask])]
    hr_bpm  = peak_hz * 60.0

    # --- HRV via peak detection ---
    min_distance = int(fps * 0.4)  # 150 bpm max
    peaks, props = sp_signal.find_peaks(
        bvp_f, distance=min_distance, prominence=0.01
    )
    hrv_rmssd = None
    if len(peaks) >= 3:
        ibi_ms = np.diff(peaks) / fps * 1000  # inter-beat intervals in ms
        # Remove physiologically implausible IBIs
        ibi_ms = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 2000)]
        if len(ibi_ms) >= 2:
            hrv_rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))

    # --- RR via respiratory signal (low-frequency BVP modulation) ---
    b2, a2 = sp_signal.butter(2, [0.15, 0.5], btype='bandpass', fs=fps)
    resp_f = sp_signal.filtfilt(b2, a2, bvp)
    freqs2, psd2 = sp_signal.welch(resp_f, fs=fps, nperseg=min(len(resp_f), 256))
    rr_mask = (freqs2 >= 0.15) & (freqs2 <= 0.5)
    rr_hz   = freqs2[rr_mask][np.argmax(psd2[rr_mask])] if rr_mask.any() else 0.25
    rr_bpm  = rr_hz * 60.0

    # --- Signal quality (SNR proxy) ---
    signal_power = psd[hr_mask].max()
    noise_power  = psd[~hr_mask].mean() + 1e-10
    snr_db = 10 * np.log10(signal_power / noise_power)
    quality = float(np.clip((snr_db - 2) / 18, 0, 1))  # 0–1

    return {
        "hr_bpm":        float(hr_bpm),
        "hrv_rmssd_ms":  hrv_rmssd,
        "rr_bpm":        float(rr_bpm),
        "quality":       quality,
    }


if __name__ == "__main__":
    # Quick smoke test
    model = FaceRPPGModel(frames=160, img_size=72, pretrained=False)
    x = torch.randn(2, 160, 3, 72, 72)
    bvp, resp = model(x)
    print(f"BVP output shape : {bvp.shape}")   # (2, 159)
    print(f"Resp output shape: {resp.shape}")  # (2, 159)

    # Loss test
    criterion = RPPGLoss()
    gt_bvp  = torch.randn(2, 159)
    gt_hr   = torch.tensor([72.0, 85.0])
    gt_resp = torch.tensor([16.0, 18.0])
    fitz    = torch.tensor([2, 5])
    losses  = criterion(bvp, resp, gt_bvp, gt_hr, gt_resp, fitz)
    print(f"Loss: {losses['loss'].item():.4f}")
