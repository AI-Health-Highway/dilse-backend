"""Risk model router.

Selects an appropriate model based on user-declared region/ethnicity, with the
ability to compute *all* applicable models for comparison (transparency).

Decision rules (v1):
  - region == 'uk' → primary QRISK3 (south asian / other ethnicities respected via QRISK3 ethnicity coeff)
  - region == 'europe' → primary SCORE2 (+QRISK3 for comparison)
  - region == 'india' / south asia → primary WHO/ISH (+QRISK3 with 'indian' ethnicity for comparison)
  - region == 'global'/unknown → primary QRISK3, secondary WHO/ISH
"""

from __future__ import annotations
from typing import Any, Dict, List

from .qrisk3 import qrisk3
from .score2 import score2
from .who_ish import who_ish


def list_models() -> List[Dict[str, str]]:
    return [
        {"key": "qrisk3", "label": "QRISK3 (UK)"},
        {"key": "score2", "label": "SCORE2 / SCORE2-OP (Europe)"},
        {"key": "who_ish", "label": "WHO/ISH (LMIC, India)"},
    ]


def route_and_run(inputs: Dict[str, Any]) -> Dict[str, Any]:
    region = (inputs.get("region") or "uk").lower()
    ethnicity = (inputs.get("ethnicity") or "").lower()

    # Determine primary + secondaries
    if region == "uk":
        primary_key = "qrisk3"
        secondaries = ["score2"] if ethnicity in ("white_or_unknown", "chinese", "other_asian", "other") else ["who_ish"]
    elif region == "europe":
        primary_key = "score2"
        secondaries = ["qrisk3"]
    elif region in ("india", "south_asia"):
        primary_key = "who_ish"
        secondaries = ["qrisk3"]
    else:
        primary_key = "qrisk3"
        secondaries = ["who_ish"]

    engines = {"qrisk3": qrisk3, "score2": score2, "who_ish": who_ish}

    primary = engines[primary_key](inputs)
    primary["role"] = "primary"

    secondary_results = []
    for k in secondaries:
        r = engines[k](inputs)
        r["role"] = "secondary"
        secondary_results.append(r)

    return {
        "primary": primary,
        "secondary": secondary_results,
        "region": region,
        "ethnicity": ethnicity,
        "selectionRationale": (
            f"Selected {primary['modelUsed']} based on region='{region}'"
            + (f", ethnicity='{ethnicity}'" if ethnicity else "")
            + ". Secondary models shown for comparison."
        ),
    }
