"""Deterministic educational heart comparison and 90-day lifestyle plan.

This uses the repository's existing QRISK3 approximation, not JBS3 or ASCVD.
Reference age is the age matching this person's risk with modifiable risk
factors set to the reference values, keeping sex and ethnicity unchanged.
"""
from __future__ import annotations

import math
from typing import Any

from risk.qrisk3 import qrisk3


def validate_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    result = dict(inputs)
    for key, low, high in (("age", 25, 84), ("sbp", 70, 250)):
        try:
            value = float(result.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"Enter {key}") from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        result[key] = value
    if result.get("sex") not in {"male", "female"}:
        raise ValueError("Select sex")
    if result.get("smoking", "non") not in {"non", "ex", "light", "moderate", "heavy"}:
        raise ValueError("Select smoking status")
    if result.get("diabetes_type", "none") not in {"none", "type1", "type2"}:
        raise ValueError("Select diabetes status")
    for key, low, high in (("bmi", 10, 80), ("total_cholesterol", 1, 20), ("hdl", 0.1, 10), ("chol_hdl_ratio", 0.5, 30), ("ldl", 0.1, 20), ("dbp", 40, 160), ("commute_hours", 0, 24)):
        if result.get(key) not in (None, ""):
            value = float(result[key])
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            result[key] = value
    if result.get("total_cholesterol") and result.get("hdl"):
        result["chol_hdl_ratio"] = result["total_cholesterol"] / result["hdl"]
    return result


def journey_summary(inputs: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    age = int(float(inputs["age"]))
    current = qrisk3(inputs)
    reference = dict(inputs, sbp=120, bmi=23, chol_hdl_ratio=3.5,
                     smoking="non", diabetes_type="none", treated_hypertension=False)
    # Nonmodifiable history stays the same; this is a modifiable-factor comparison.
    candidates = [(a, qrisk3(dict(reference, age=a))["score"]) for a in range(25, 85)]
    matches = sorted(candidates, key=lambda pair: (abs(pair[1] - current["score"]), abs(pair[0] - age)))
    heart_age = matches[0][0]
    range_note = "above" if current["score"] > candidates[-1][1] else "below" if current["score"] < candidates[0][1] else None
    factors = []
    for key, label, active, value in (
        ("sbp", "Blood pressure", float(inputs["sbp"]) > 120, f"{inputs['sbp']:g} mmHg"),
        ("smoking", "Smoking", inputs.get("smoking", "non") not in {"non", "ex"}, inputs.get("smoking")),
        ("diabetes_type", "Diabetes", inputs.get("diabetes_type", "none") != "none", inputs.get("diabetes_type")),
        ("bmi", "BMI", float(inputs.get("bmi") or 23) > 25, inputs.get("bmi")),
        ("chol_hdl_ratio", "Cholesterol ratio", float(inputs.get("chol_hdl_ratio") or 3.5) > 3.5, inputs.get("chol_hdl_ratio")),
        ("family_history", "Family history", bool(inputs.get("family_history")), "Reported"),
        ("ckd", "Kidney disease", bool(inputs.get("ckd")), "Reported"),
    ):
        if active:
            factors.append({"key": key, "label": label, "value": value})
    first = ["Build towards 150 minutes of moderate movement per week, at a comfortable pace.",
             "Replace sugary drinks with water and add vegetables to meals.",
             "Keep a regular sleep schedule."]
    if float(inputs["sbp"]) >= 130 or inputs.get("treated_hypertension"):
        first.append("Record cuff blood pressure 3 times a week and review the log with a clinician.")
    if inputs.get("smoking", "non") not in {"non", "ex"}:
        first.append("Arrange support for stopping tobacco use.")
    if inputs.get("diabetes_type", "none") != "none":
        first.append("Review diabetes monitoring and your existing care plan with your clinician.")
    if risk["primary"]["band"] in {"high", "very_high"} or float(inputs["sbp"]) >= 160:
        first.append("Arrange a clinical consultation to review your risk factors.")
    second = ["Maintain regular walking and add strength activity on 2 days a week if suitable.",
              "Build meals around vegetables, whole grains and pulses; reduce excess salt.",
              "Review your progress and repeat the questionnaire on day 60."]
    if float(inputs.get("commute_hours") or 0) > 1:
        second.append("Break up commute and desk sitting with regular movement; consider flexible work where possible.")
    return {
        "calendarAge": age, "heartAge": heart_age, "difference": heart_age - age,
        "heartAgeModel": "QRISK3-approx reference-age comparison", "rangeNote": range_note,
        "explanation": "Educational risk-equivalent age, not a measured biological age. Uses an approximate calculator; plaque and stiffness are not measured by a camera scan.",
        "factors": factors,
        "currentProfile": {key: inputs.get(key) for key in ("sbp", "dbp", "bmi", "ldl", "ethnicity", "commute_hours")},
        "healthyTarget": {"heartAge": age, "description": "Work towards a healthier risk profile; a specific heart-age reduction is not guaranteed."},
        "plan": [
            {"day": 30, "range": "Day 1–30", "title": "Stabilize", "actions": first},
            {"day": 60, "range": "Day 31–60", "title": "Rebuild", "actions": second},
            {"day": 90, "range": "Day 61–90", "title": "Embed", "actions": ["Keep the activity and food habits that worked for you.", "Repeat cuff or lab measurements where your clinician recommends them.", "Repeat Heart Risk + Heart Age on day 90 and compare with your baseline."]},
        ],
        "dailyTasks": ["Take a comfortable 20-minute walk if suitable.", "Make half your plate vegetables.", "Break up prolonged sitting each hour."],
        "clinicianNote": "Lifestyle plan only. Medicines and individual exercise limits should be discussed with your doctor.",
        "sources": ["https://www.who.int/europe/news-room/fact-sheets/item/physical-activity"],
    }


def validate_raw_measurements(raw: Any) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict) or raw.get("source") not in {"ppg", "rppg"}:
        raise ValueError("Invalid raw measurement source")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) > 900:
        raise ValueError("Raw measurements must contain at most 900 samples")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Invalid pulse sample")
        for key in ("t", "r", "g", "b"):
            value = sample.get(key)
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("Invalid pulse sample")
