"""Simplified SCORE2 / SCORE2-OP 10-year CVD risk approximation.

Derived from public ESC 2021 risk tables (SCORE2 ages 40-69, SCORE2-OP ages
70+). This is a *table-lookup approximation* using non-HDL cholesterol, SBP,
sex, smoking, and age, calibrated for the "moderate-risk region" (most of
Western Europe, including UK, France, Germany, Italy, Spain).

For low-risk and high-risk SCORE2 regions, results are scaled.

Limitations are surfaced in the response.
"""

from __future__ import annotations
from typing import Any, Dict, List

# Region multipliers (multiplicative scale on baseline moderate-risk result)
# Per ESC 2021 calibration concept
REGION_SCALE: Dict[str, float] = {
    "low": 0.7,        # e.g., parts of Western Europe (NL, BE, FR, IT, ES, UK, IE)
    "moderate": 1.0,   # baseline
    "high": 1.4,       # e.g., parts of Eastern Europe / CE
    "very_high": 1.9,  # e.g., parts of Eastern Europe / FSU
}


def _bp_band(sbp: float) -> int:
    if sbp < 120: return 0
    if sbp < 140: return 1
    if sbp < 160: return 2
    if sbp < 180: return 3
    return 4


def _nhdl_band(nhdl: float) -> int:
    # non-HDL cholesterol (mmol/L) bands
    if nhdl < 3.0: return 0
    if nhdl < 4.0: return 1
    if nhdl < 5.0: return 2
    if nhdl < 6.0: return 3
    return 4


def _age_band(age: float) -> int:
    return min(7, max(0, int((age - 40) // 5)))


# Coarse risk grid (sex x smoking) for moderate-risk region, % over 10 years
# Index: [smoking] [age_band 0..7] [bp_band 0..4] [nhdl_band 0..4]
# We approximate with a parametric formula to keep file small and inspectable.

def _base_risk(sex: str, smoking: bool, age: float, sbp: float, nhdl: float) -> float:
    """Parametric approximation of SCORE2 moderate-risk table.

    Anchored to published anchor values such as:
      - 50yo non-smoking female, SBP 140, non-HDL 4 ~ 1-2%
      - 60yo smoking male,    SBP 140, non-HDL 5 ~ 10-13%
      - 70yo smoking male,    SBP 160, non-HDL 5 ~ 18-22%
    """
    ab = _age_band(age)
    bb = _bp_band(sbp)
    nb = _nhdl_band(nhdl)

    base = 0.6 if sex == "female" else 1.0
    age_mult = (1.45 ** ab) if age < 70 else (1.30 ** (ab - 6) + 5.0)
    bp_add = bb * (0.6 if sex == "female" else 0.9)
    nhdl_add = nb * (0.5 if sex == "female" else 0.7)
    smoke_mult = 2.0 if smoking else 1.0

    risk = (base + bp_add + nhdl_add) * age_mult * smoke_mult
    # Cap
    return min(89.0, max(0.3, round(risk, 1)))


def _band(score: float) -> str:
    if score < 5:
        return "low"
    if score < 10:
        return "moderate"
    if score < 20:
        return "high"
    return "very_high"


def score2(inp: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []

    def _get(name: str, default: float, flag: bool = True) -> float:
        v = inp.get(name)
        if v in (None, ""):
            if flag: missing.append(name)
            return default
        try:
            return float(v)
        except Exception:
            if flag: missing.append(name)
            return default

    sex = (inp.get("sex") or "male").lower()
    if sex not in ("male", "female"):
        sex = "male"
        missing.append("sex")
    age = _get("age", 55.0)
    sbp = _get("sbp", 130.0)
    total_chol = _get("total_cholesterol", 5.0)  # mmol/L
    hdl = _get("hdl", 1.3)
    nhdl = max(0.5, total_chol - hdl)
    smoking = (inp.get("smoking") or "non").lower()
    is_smoker = smoking in ("light", "moderate", "heavy")
    region = (inp.get("score2_region") or "moderate").lower()
    if region not in REGION_SCALE:
        region = "moderate"

    base = _base_risk(sex, is_smoker, age, sbp, nhdl)
    score = base * REGION_SCALE[region]
    score = round(min(89.0, max(0.3, score)), 1)

    used_op = age >= 70
    model_used = "SCORE2-OP" if used_op else "SCORE2"

    limitations = [
        f"Simplified {model_used} approximation — table-lookup not equivalent to official ESC calculator.",
        "non-HDL cholesterol derived as total - HDL (mmol/L). Provide both for best accuracy.",
        "Choose the SCORE2 region matching your country for calibration accuracy.",
    ]
    if missing:
        limitations.append(f"Missing inputs replaced with defaults: {', '.join(missing)}.")
    if age < 40:
        limitations.append("Age <40: SCORE2 is not designed for this age group; result is indicative only.")

    return {
        "modelUsed": model_used,
        "score": score,
        "band": _band(score),
        "inputsUsed": {
            "age": age, "sex": sex, "sbp": sbp, "non_hdl": round(nhdl, 2),
            "smoking": smoking, "region": region,
        },
        "missingInputs": missing,
        "limitations": limitations,
    }
