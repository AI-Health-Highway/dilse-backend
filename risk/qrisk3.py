"""Simplified QRISK3-style 10-year CVD risk approximation.

This is a pragmatic, *educational* re-implementation inspired by the public
QRISK3 equations (Hippisley-Cox et al., BMJ 2017). It does not reproduce the
full coefficient set or fractional polynomial transforms required for a
clinically validated QRISK3 result.

Users MUST be informed (via the API response `limitations` field) that the
result is directional and not equivalent to the official QRISK3 calculator at
qrisk.org.

Inputs (dict):
    age (years), sex ('male'|'female'), ethnicity (key, see ETHNICITY_KEYS),
    smoking ('non'|'ex'|'light'|'moderate'|'heavy'),
    sbp (mmHg), bmi (kg/m^2), townsend (deprivation, -7 to +11; 0 default),
    chol_hdl_ratio (e.g., 4.5),
    diabetes_type ('none'|'type1'|'type2'),
    family_history (bool), treated_hypertension (bool),
    ckd (bool), af (bool), ra (bool), migraine (bool), sle (bool),
    severe_mental_illness (bool), corticosteroids (bool), atypical_antipsychotics (bool),
    erectile_dysfunction (bool)

Output (dict):
    score (float, percent, 0..99),
    band ('low'|'moderate'|'high'),
    inputsUsed (dict),
    missingInputs (list[str]),
    limitations (list[str])
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

ETHNICITY_KEYS = [
    "white_or_unknown", "indian", "pakistani", "bangladeshi",
    "other_asian", "black_caribbean", "black_african", "chinese", "other",
]

# Ethnicity adjustments (multiplicative on linear predictor, simplified)
# Approximation: South Asian groups carry higher CVD risk; East Asian lower.
ETHNICITY_ADJ: Dict[str, float] = {
    "white_or_unknown": 0.0,
    "indian": 0.30,
    "pakistani": 0.40,
    "bangladeshi": 0.45,
    "other_asian": 0.20,
    "black_caribbean": 0.15,
    "black_african": 0.10,
    "chinese": -0.10,
    "other": 0.0,
}

SMOKING_ADJ: Dict[str, Dict[str, float]] = {
    # additive on linear predictor by sex
    "male":   {"non": 0.0, "ex": 0.18, "light": 0.36, "moderate": 0.55, "heavy": 0.72},
    "female": {"non": 0.0, "ex": 0.22, "light": 0.40, "moderate": 0.60, "heavy": 0.80},
}

# Coefficients (educational simplification derived from QRISK3-like effects)
# Male
MALE = {
    "age": 0.0735,     # per year above 35
    "sbp": 0.0133,     # per mmHg above 130
    "chol_hdl": 0.1841,  # per unit ratio above 4
    "bmi": 0.0283,     # per unit above 27
    "townsend": 0.0332,  # per Townsend point
    "diabetes_t1": 1.10,
    "diabetes_t2": 0.85,
    "family_history": 0.51,
    "treated_htn": 0.55,
    "ckd": 0.45,
    "af": 0.95,
    "ra": 0.21,
    "migraine": 0.06,
    "sle": 0.30,
    "severe_mental": 0.18,
    "corticosteroids": 0.20,
    "atyp_aps": 0.10,
    "erectile_dys": 0.20,
    "baseline_surv": 0.977,  # baseline 10-yr survival ~ produces realistic %
}
FEMALE = {
    "age": 0.0790,
    "sbp": 0.0102,
    "chol_hdl": 0.1532,
    "bmi": 0.0181,
    "townsend": 0.0292,
    "diabetes_t1": 1.30,
    "diabetes_t2": 0.95,
    "family_history": 0.52,
    "treated_htn": 0.58,
    "ckd": 0.50,
    "af": 1.00,
    "ra": 0.21,
    "migraine": 0.08,
    "sle": 0.40,
    "severe_mental": 0.18,
    "corticosteroids": 0.22,
    "atyp_aps": 0.10,
    "erectile_dys": 0.0,
    "baseline_surv": 0.985,
}


def _band(score: float) -> str:
    if score < 10:
        return "low"
    if score < 20:
        return "moderate"
    return "high"


def qrisk3(inp: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    sex = (inp.get("sex") or "").lower()
    if sex not in ("male", "female"):
        sex = "male"  # default; flagged below
        missing.append("sex")
    C = MALE if sex == "male" else FEMALE

    def _get(name: str, default: float, flag: bool = True) -> float:
        v = inp.get(name)
        if v is None or v == "":
            if flag:
                missing.append(name)
            return default
        try:
            return float(v)
        except Exception:
            if flag:
                missing.append(name)
            return default

    age = _get("age", 50.0)
    sbp = _get("sbp", 125.0)
    bmi = _get("bmi", 26.0)
    chol_hdl = _get("chol_hdl_ratio", 4.0)
    townsend = _get("townsend", 0.0, flag=False)

    ethnicity = (inp.get("ethnicity") or "white_or_unknown").lower()
    if ethnicity not in ETHNICITY_KEYS:
        ethnicity = "white_or_unknown"
    smoking = (inp.get("smoking") or "non").lower()
    if smoking not in SMOKING_ADJ["male"]:
        smoking = "non"

    diabetes_type = (inp.get("diabetes_type") or "none").lower()

    lp = 0.0
    lp += C["age"] * max(0.0, age - 35.0)
    lp += C["sbp"] * max(0.0, sbp - 130.0)
    lp += C["chol_hdl"] * max(0.0, chol_hdl - 4.0)
    lp += C["bmi"] * max(0.0, bmi - 27.0)
    lp += C["townsend"] * townsend
    lp += SMOKING_ADJ[sex][smoking]
    lp += ETHNICITY_ADJ[ethnicity]

    if diabetes_type == "type1":
        lp += C["diabetes_t1"]
    elif diabetes_type == "type2":
        lp += C["diabetes_t2"]

    boolean_flags: List[Tuple[str, str]] = [
        ("family_history", "family_history"),
        ("treated_hypertension", "treated_htn"),
        ("ckd", "ckd"),
        ("af", "af"),
        ("ra", "ra"),
        ("migraine", "migraine"),
        ("sle", "sle"),
        ("severe_mental_illness", "severe_mental"),
        ("corticosteroids", "corticosteroids"),
        ("atypical_antipsychotics", "atyp_aps"),
    ]
    for key, coef_key in boolean_flags:
        if bool(inp.get(key)):
            lp += C[coef_key]
    if sex == "male" and bool(inp.get("erectile_dysfunction")):
        lp += C["erectile_dys"]

    # Convert LP to 10-yr risk via Cox-like baseline survival
    s10 = C["baseline_surv"]
    risk = 1.0 - math.pow(s10, math.exp(lp))
    score = max(0.5, min(99.0, risk * 100.0))
    score = round(score, 1)

    limitations = [
        "Educational approximation of QRISK3 — not equivalent to the official calculator at qrisk.org.",
        "For UK populations the official QRISK3 is recommended; for non-UK populations consider SCORE2 (Europe) or WHO/ISH (LMIC).",
        "Self-reported inputs reduce reliability. Cuff BP and lab cholesterol improve accuracy.",
    ]
    if missing:
        limitations.append(
            f"Missing inputs replaced with population defaults: {', '.join(missing)}."
        )

    return {
        "modelUsed": "QRISK3-approx",
        "score": score,
        "band": _band(score),
        "inputsUsed": {
            "age": age, "sex": sex, "sbp": sbp, "bmi": bmi,
            "chol_hdl_ratio": chol_hdl, "townsend": townsend,
            "ethnicity": ethnicity, "smoking": smoking,
            "diabetes_type": diabetes_type,
        },
        "missingInputs": missing,
        "limitations": limitations,
    }
