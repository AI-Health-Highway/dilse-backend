"""Simplified WHO/ISH 10-year CVD risk chart (no cholesterol required).

Based on the WHO/ISH risk prediction charts (2007, updated 2019). These were
designed for low- and middle-income countries where cholesterol testing may
not be routinely available, and instead categorize 10-yr CVD risk into bands
based on: WHO sub-region, age, sex, smoking status, SBP, diabetes, and
optionally total cholesterol when known.

The official charts report risk as bands (e.g., <10%, 10-20%, 20-30%, 30-40%,
>=40%). This module returns a midpoint of the matching band as a continuous
score plus the categorical band.

The `who_subregion` accepted values are e.g., 'sear_d' (South-East Asia D —
includes India), 'sear_b', 'amr_a', etc. If not provided, defaults to 'sear_d'
(India) since the user community is India-focused.
"""

from __future__ import annotations
from typing import Any, Dict, List

# Risk band midpoints used as the numeric score returned
BAND_MIDPOINTS = {
    "<10": 5.0,
    "10-20": 15.0,
    "20-30": 25.0,
    "30-40": 35.0,
    ">=40": 45.0,
}


def _band_label(score: float) -> str:
    if score < 10:
        return "low"
    if score < 20:
        return "moderate"
    if score < 30:
        return "high"
    return "very_high"


def _approx_who_band(age: float, sex: str, smoking: bool, sbp: float, diabetes: bool, total_chol: float | None) -> str:
    """Heuristic that mimics the WHO/ISH chart for SEAR-D (India) without chol.

    Risk increases monotonically with: age, male sex, smoking, SBP, diabetes,
    and (if provided) total cholesterol.
    """
    # Start with a numeric risk index
    idx = 0.0
    # Age bands roughly per WHO/ISH (40, 50, 60, 70)
    if age >= 70: idx += 4.0
    elif age >= 60: idx += 3.0
    elif age >= 50: idx += 2.0
    elif age >= 40: idx += 1.0

    if sex == "male": idx += 0.8

    if smoking: idx += 1.5

    if sbp >= 180: idx += 3.0
    elif sbp >= 160: idx += 2.0
    elif sbp >= 140: idx += 1.2
    elif sbp >= 120: idx += 0.3

    if diabetes: idx += 1.6

    if total_chol is not None:
        if total_chol >= 8.0: idx += 1.4
        elif total_chol >= 6.0: idx += 0.8
        elif total_chol >= 5.0: idx += 0.3

    if idx < 2.5:
        return "<10"
    if idx < 4.5:
        return "10-20"
    if idx < 6.5:
        return "20-30"
    if idx < 8.5:
        return "30-40"
    return ">=40"


def who_ish(inp: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []

    def _get(name: str, default, flag=True):
        v = inp.get(name)
        if v in (None, ""):
            if flag: missing.append(name)
            return default
        return v

    sex = (inp.get("sex") or "male").lower()
    if sex not in ("male", "female"):
        sex = "male"
        missing.append("sex")
    age = float(_get("age", 55.0))
    sbp = float(_get("sbp", 130.0))
    smoking = (inp.get("smoking") or "non").lower()
    is_smoker = smoking in ("light", "moderate", "heavy")
    diabetes = (inp.get("diabetes_type") or "none").lower() in ("type1", "type2")
    total_chol_raw = inp.get("total_cholesterol")
    total_chol = float(total_chol_raw) if total_chol_raw not in (None, "") else None
    subregion = (inp.get("who_subregion") or "sear_d").lower()

    band = _approx_who_band(age, sex, is_smoker, sbp, diabetes, total_chol)
    score = BAND_MIDPOINTS[band]

    limitations = [
        "Simplified WHO/ISH chart approximation — for screening only, not equivalent to clinical lookup.",
        f"WHO sub-region used: {subregion}. Default 'sear_d' targets India and similar populations.",
        "Cholesterol is optional; when omitted, risk is estimated from age/sex/SBP/smoking/diabetes only.",
    ]
    if missing:
        limitations.append(f"Missing inputs replaced with defaults: {', '.join(missing)}.")

    return {
        "modelUsed": "WHO/ISH",
        "score": score,
        "band": _band_label(score),
        "rawBand": band,
        "inputsUsed": {
            "age": age, "sex": sex, "sbp": sbp,
            "smoking": smoking, "diabetes": diabetes,
            "total_cholesterol": total_chol,
            "who_subregion": subregion,
        },
        "missingInputs": missing,
        "limitations": limitations,
    }
