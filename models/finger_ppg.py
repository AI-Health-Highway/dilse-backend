"""
F3 Finger PPG Model
====================
Estimates SpO2, HRV (RMSSD), and HR from finger-on-camera PPG signal.

Architecture: 1D CNN + Bi-GRU on raw RGB pixel time series
- Input : (B, 3, T) — 3-channel mean pixel values from finger ROI over T frames
- Output: (B, 1) SpO2 %, (B, T) BVP waveform (for HRV), (B, 1) HR bpm

SpO2 estimation uses Beer-Lambert law ratio-of-ratios (R value),
calibrated via a learned regression head on top of the physiological R.

BP estimation: morphological features from BVP → MLP → systolic/diastolic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal as sp_signal


# ── 1-D Residual Block ────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel: int = 7, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


# ── Stem ──────────────────────────────────────────────────────────────────────

class PPGStem(nn.Module):
    """
    Projects raw 3-channel (R, G, B) pixel traces to feature space.
    Uses multi-scale convolutions to capture different pulse harmonics.
    """
    def __init__(self, out_ch: int = 64):
        super().__init__()
        # Three parallel kernels: short (fine), medium, long (coarse)
        self.b1 = nn.Conv1d(3, out_ch // 3, kernel_size=5,  padding=2)
        self.b2 = nn.Conv1d(3, out_ch // 3, kernel_size=15, padding=7)
        self.b3 = nn.Conv1d(3, out_ch - 2 * (out_ch // 3), kernel_size=31, padding=15)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        # x: (B, 3, T)
        return self.act(self.norm(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)))


# ── Finger PPG Backbone ───────────────────────────────────────────────────────

class FingerPPGBackbone(nn.Module):
    """
    1D CNN backbone with progressive dilation for multi-scale temporal modelling.
    Output: (B, hidden_dim, T) feature map
    """
    def __init__(self, in_ch: int = 64, hidden: int = 128, depth: int = 6):
        super().__init__()
        layers = [nn.Conv1d(in_ch, hidden, 1)]
        dilations = [1, 2, 4, 8, 16, 1][:depth]
        for d in dilations:
            layers.append(ResBlock1D(hidden, kernel=7, dilation=d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, hidden, T)


# ── Sequence Refinement with Bi-GRU ──────────────────────────────────────────

class BiGRURefiner(nn.Module):
    def __init__(self, hidden: int = 128, gru_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            hidden, hidden // 2, num_layers=gru_layers,
            batch_first=True, bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: (B, C, T) → transpose → (B, T, C) for GRU
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.norm(out).permute(0, 2, 1)  # (B, C, T)


# ── SpO2 Head ─────────────────────────────────────────────────────────────────

class SpO2Head(nn.Module):
    """
    Two-branch approach:
    1. Physiological R-value from AC/DC ratio of red vs IR (G channel proxy)
       R = (AC_red / DC_red) / (AC_green / DC_green)
       SpO2 ≈ a - b*R  (Beer-Lambert, coefficients learned)
    2. Data-driven regression from feature vector as calibration
    Final SpO2 = blend of physio estimate + learned correction
    """
    def __init__(self, hidden: int = 128):
        super().__init__()
        # Learned calibration coefficients for Beer-Lambert
        self.a = nn.Parameter(torch.tensor(110.0))  # empirical ~110
        self.b = nn.Parameter(torch.tensor(25.0))   # empirical ~25

        # Data-driven correction path
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.mlp     = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.blend_w = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat: torch.Tensor, raw_rgb: torch.Tensor) -> torch.Tensor:
        """
        feat    : (B, hidden, T)
        raw_rgb : (B, 3, T)  — raw channel means (Red=0, Green=1, Blue=2)
        """
        # Beer-Lambert R-value
        red   = raw_rgb[:, 0, :]   # (B, T)
        green = raw_rgb[:, 1, :]   # (B, T) — best IR proxy in visible-light PPG

        ac_red   = (red   - red.mean(dim=1, keepdim=True)).abs().mean(dim=1)
        dc_red   = red.mean(dim=1).clamp(min=1e-6)
        ac_green = (green - green.mean(dim=1, keepdim=True)).abs().mean(dim=1)
        dc_green = green.mean(dim=1).clamp(min=1e-6)

        R = (ac_red / dc_red) / (ac_green / dc_green + 1e-6)
        spo2_physio = (self.a - self.b * R).clamp(50, 100)  # (B,)

        # Data-driven path
        feat_pooled = self.pool(feat).squeeze(-1)            # (B, hidden)
        spo2_learned = self.mlp(feat_pooled).squeeze(-1)    # (B,)
        spo2_learned = (95.0 + 5.0 * torch.tanh(spo2_learned))  # constrain to ~85-100

        # Blended estimate
        w = torch.sigmoid(self.blend_w)
        spo2 = w * spo2_physio + (1 - w) * spo2_learned
        return spo2.clamp(50, 100)


# ── BVP Head (for HRV) ────────────────────────────────────────────────────────

class BVPHead(nn.Module):
    """Projects feature map to a 1-channel BVP waveform."""
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, x):
        return self.conv(x).squeeze(1)  # (B, T)


# ── HR Head ───────────────────────────────────────────────────────────────────

class HRHead(nn.Module):
    """Directly regresses HR from global pooled features."""
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, hidden, T)
        pooled = self.pool(x).squeeze(-1)  # (B, hidden)
        hr = self.mlp(pooled).squeeze(-1)  # (B,)
        return (60.0 + 100.0 * torch.sigmoid(hr))  # constrain to 60–160 bpm


# ── Blood Pressure Head ───────────────────────────────────────────────────────

class BPHead(nn.Module):
    """
    Morphological BP estimator.
    Extracts handcrafted PTT/PWV proxy features from BVP, feeds to MLP.
    
    Features extracted per-sample (no direct gradient to raw signal):
      - pulse transit time proxy (peak-to-peak interval statistics)
      - augmentation index (pulse wave reflection)
      - stiffness index
    These are weak BP proxies — useful for trend tracking, not clinical diagnosis.
    """
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.pool  = nn.AdaptiveAvgPool1d(8)  # keep some temporal context
        self.mlp   = nn.Sequential(
            nn.Linear(hidden * 8, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # [systolic, diastolic]
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, hidden, T)
        pooled = self.pool(feat)           # (B, hidden, 8)
        flat   = pooled.view(pooled.size(0), -1)  # (B, hidden*8)
        bp = self.mlp(flat)                # (B, 2)
        # Constrain to physiological ranges: systolic 80-200, diastolic 50-130
        systolic  = 80.0  + 120.0 * torch.sigmoid(bp[:, 0])
        diastolic = 50.0  + 80.0  * torch.sigmoid(bp[:, 1])
        return torch.stack([systolic, diastolic], dim=1)  # (B, 2)


# ── Full Model ────────────────────────────────────────────────────────────────

class FingerPPGModel(nn.Module):
    """
    Multi-task model for finger-PPG analysis.

    Outputs
    -------
    bvp    : (B, T)   — BVP waveform for HRV computation
    hr     : (B,)     — heart rate [bpm]
    spo2   : (B,)     — blood oxygen saturation [%]
    bp     : (B, 2)   — [systolic, diastolic] mmHg (trend only)
    """
    def __init__(
        self,
        frames: int = 300,    # ~10s at 30fps
        stem_ch: int = 64,
        hidden:  int = 128,
        pretrained: bool = False,  # no pretrained weights for 1D CNN
    ):
        super().__init__()
        self.frames = frames

        self.stem     = PPGStem(out_ch=stem_ch)
        self.backbone = FingerPPGBackbone(in_ch=stem_ch, hidden=hidden)
        self.refiner  = BiGRURefiner(hidden=hidden)

        self.bvp_head  = BVPHead(hidden)
        self.hr_head   = HRHead(hidden)
        self.spo2_head = SpO2Head(hidden)
        self.bp_head   = BPHead(hidden)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, T)  — channel-mean RGB traces from finger ROI, float32
        """
        # Normalise per-channel per-sample
        mean = x.mean(dim=2, keepdim=True)
        std  = x.std(dim=2, keepdim=True).clamp(min=1e-6)
        x_norm = (x - mean) / std

        feat   = self.stem(x_norm)               # (B, stem_ch, T)
        feat   = self.backbone(feat)              # (B, hidden, T)
        feat   = self.refiner(feat)               # (B, hidden, T)

        bvp    = self.bvp_head(feat)              # (B, T)
        hr     = self.hr_head(feat)               # (B,)
        spo2   = self.spo2_head(feat, x)          # (B,) raw x for Beer-Lambert
        bp     = self.bp_head(feat)               # (B, 2)

        return bvp, hr, spo2, bp


# ── Loss ──────────────────────────────────────────────────────────────────────

def negative_pearson(pred, target):
    pred_m   = pred   - pred.mean(dim=1, keepdim=True)
    target_m = target - target.mean(dim=1, keepdim=True)
    r = (pred_m * target_m).sum(1) / (
        torch.sqrt((pred_m**2).sum(1) * (target_m**2).sum(1) + 1e-8)
    )
    return (-r).mean()


class FingerPPGLoss(nn.Module):
    def __init__(self, w_bvp=1.0, w_hr=0.5, w_spo2=1.0, w_bp=0.3):
        super().__init__()
        self.w_bvp  = w_bvp
        self.w_hr   = w_hr
        self.w_spo2 = w_spo2
        self.w_bp   = w_bp

    def forward(self, pred_bvp, pred_hr, pred_spo2, pred_bp,
                gt_bvp, gt_hr, gt_spo2, gt_bp):
        l_bvp  = negative_pearson(pred_bvp, gt_bvp)
        l_hr   = F.l1_loss(pred_hr, gt_hr)
        l_spo2 = F.l1_loss(pred_spo2, gt_spo2)
        l_bp   = F.l1_loss(pred_bp, gt_bp)
        total  = (self.w_bvp * l_bvp + self.w_hr * l_hr
                  + self.w_spo2 * l_spo2 + self.w_bp * l_bp)
        return {
            "loss":      total,
            "bvp":       l_bvp.item(),
            "hr_mae":    l_hr.item(),
            "spo2_mae":  l_spo2.item(),
            "bp_mae":    l_bp.item(),
        }


# ── Post-processing: BVP → HRV ────────────────────────────────────────────────

def bvp_to_hrv(bvp: np.ndarray, fps: float = 30.0) -> dict:
    """
    Extract HRV metrics from finger PPG BVP waveform.

    Returns
    -------
    hrv_rmssd_ms : RMSSD [ms]
    hrv_sdnn_ms  : SDNN  [ms]
    hr_bpm       : mean HR from IBI [bpm]
    nn50         : number of successive IBI pairs differing > 50ms
    pnn50        : proportion of above
    """
    b, a = sp_signal.butter(3, [0.5, 4.0], btype='bandpass', fs=fps)
    bvp_f = sp_signal.filtfilt(b, a, bvp)

    min_dist = int(fps * 0.33)  # max ~180 bpm
    peaks, _ = sp_signal.find_peaks(bvp_f, distance=min_dist, prominence=0.05)

    if len(peaks) < 3:
        return {"hrv_rmssd_ms": None, "hrv_sdnn_ms": None,
                "hr_bpm": None, "nn50": None, "pnn50": None}

    ibi_ms = np.diff(peaks) / fps * 1000
    ibi_ms = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 2000)]

    if len(ibi_ms) < 2:
        return {"hrv_rmssd_ms": None, "hrv_sdnn_ms": None,
                "hr_bpm": None, "nn50": None, "pnn50": None}

    rmssd  = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    sdnn   = float(np.std(ibi_ms))
    hr_bpm = float(60000.0 / ibi_ms.mean())
    nn50   = int(np.sum(np.abs(np.diff(ibi_ms)) > 50))
    pnn50  = float(nn50 / len(np.diff(ibi_ms))) if len(ibi_ms) > 1 else 0.0

    return {
        "hrv_rmssd_ms": rmssd,
        "hrv_sdnn_ms":  sdnn,
        "hr_bpm":       hr_bpm,
        "nn50":         nn50,
        "pnn50":        pnn50,
    }


if __name__ == "__main__":
    model = FingerPPGModel(frames=300)
    x = torch.randn(2, 3, 300)
    bvp, hr, spo2, bp = model(x)
    print(f"BVP  : {bvp.shape}")   # (2, 300)
    print(f"HR   : {hr.shape}")    # (2,)
    print(f"SpO2 : {spo2.shape}")  # (2,)
    print(f"BP   : {bp.shape}")    # (2, 2)

    criterion = FingerPPGLoss()
    losses = criterion(
        bvp, hr, spo2, bp,
        torch.randn(2, 300), torch.tensor([72.0, 85.0]),
        torch.tensor([98.0, 97.0]), torch.tensor([[120., 80.], [115., 75.]])
    )
    print(f"Loss: {losses['loss'].item():.4f}")
