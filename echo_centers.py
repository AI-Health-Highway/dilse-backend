"""Diagnostic-partner brand cards for the DilSay echo-booking flow.

We no longer maintain a static list of branches — instead we surface the three
partner brands and delegate 'find the closest one to me' to Google Maps. This
gives users always-fresh data (opening hours, phones, directions, reviews)
without us needing a Google Places API key or a costly per-lookup billing.

Frontend does:
    navigator.geolocation.getCurrentPosition() → lat/lng
    open `https://www.google.com/maps/search/?api=1&query=<brand>+near+<lat>,<lng>`
"""
from __future__ import annotations
from typing import Dict, List, Any

BRAND_CARDS: List[Dict[str, Any]] = [
    {
        "id": "aarthi",
        "brand": "aarthi",
        "displayName": "Aarthi Scans & Labs",
        "tagline": "Strong presence across South & PAN India",
        "supportPhone": "+91-44-4297-4444",
        "website": "https://www.aarthiscan.com/",
        "mapsQuery": "Aarthi Scans and Labs",
        "logoInitial": "A",
        "accentColor": "#E8445A",
        "echoAvailable": True,
    },
    {
        "id": "lalpath",
        "brand": "lalpath",
        "displayName": "Dr. Lal PathLabs",
        "tagline": "Nationwide network — strong in North & West India",
        "supportPhone": "+91-11-4988-5050",
        "website": "https://www.lalpathlabs.com/",
        "mapsQuery": "Dr Lal PathLabs",
        "logoInitial": "L",
        "accentColor": "#F5C87A",
        "echoAvailable": True,
    },
    {
        "id": "clumax",
        "brand": "clumax",
        "displayName": "Clumax Diagnostics",
        "tagline": "Karnataka HQ · strong Bengaluru coverage",
        "supportPhone": "+91-80-4212-0000",
        "website": "https://clumax.in/",
        "mapsQuery": "Clumax Diagnostics",
        "logoInitial": "C",
        "accentColor": "#7DD3A0",
        "echoAvailable": True,
    },
]


def list_partners() -> List[Dict[str, Any]]:
    return BRAND_CARDS


# Backwards-compat helpers so any lingering callers keep working
BRAND_META = {c["brand"]: c for c in BRAND_CARDS}


def find_centers(city: str | None = None, brand: str | None = None, limit: int = 50):
    rows = BRAND_CARDS
    if brand:
        rows = [c for c in rows if c["brand"] == brand.lower()]
    return rows[:max(1, min(limit, 100))]


def list_cities() -> List[str]:
    # Cities are irrelevant now — Google Maps handles proximity.
    return []
