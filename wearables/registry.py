"""Declarative registry of wearable data providers.

Each entry is pure configuration — no network, no side effects — so the whole
registry can be imported and asserted against in tests.

Three tiers, because the constraint is structural and worth naming in code:

  cloud   The vendor runs an OAuth2 REST API. A *website* can connect these.
          No phone app required. This is Garmin, Oura, Whoop, Google Health.

  native  The data lives on the phone and the platform exposes no web API at
          all. Apple Health (HealthKit) and Google Health Connect. The only
          honest web paths are a manual export import or shipping a native app.

  local   The browser itself is the transport: Web Bluetooth, manual entry.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# ── Metric vocabulary ────────────────────────────────────────────────────
# Every adapter normalizes into exactly these keys. Adding a metric means
# adding it here first, so a provider can never invent a field silently.
METRICS = ("resting_hr", "hrv_ms", "spo2", "steps", "sleep_min", "resp_rate")


PROVIDERS: Dict[str, Dict[str, Any]] = {
    # ── Oura ─────────────────────────────────────────────────────────────
    # Self-serve app registration. Personal Access Tokens were deprecated in
    # Dec 2025, so OAuth2 is the only remaining path.
    "oura": {
        "id": "oura",
        "name": "Oura Ring",
        "tier": "cloud",
        "authorize_url": "https://cloud.ouraring.com/oauth/authorize",
        "token_url": "https://api.ouraring.com/oauth/token",
        "api_base": "https://api.ouraring.com/v2",
        "scopes": ["daily", "heartrate", "spo2", "personal"],
        "scope_sep": " ",
        "pkce": False,
        "token_auth": "basic",  # client_id/secret via HTTP Basic
        "metrics": ["resting_hr", "hrv_ms", "spo2", "steps", "sleep_min", "resp_rate"],
        "client_id_env": "OURA_CLIENT_ID",
        "client_secret_env": "OURA_CLIENT_SECRET",
        "signup_url": "https://cloud.ouraring.com/oauth/applications",
        "approval": "instant",
    },
    # ── WHOOP ────────────────────────────────────────────────────────────
    # `offline` scope is mandatory to receive a refresh token. WHOOP rotates
    # refresh tokens on every use — the adapter layer must persist the new one.
    "whoop": {
        "id": "whoop",
        "name": "WHOOP",
        "tier": "cloud",
        "authorize_url": "https://api.prod.whoop.com/oauth/oauth2/auth",
        "token_url": "https://api.prod.whoop.com/oauth/oauth2/token",
        "api_base": "https://api.prod.whoop.com/developer",
        "scopes": ["read:recovery", "read:sleep", "read:cycles", "read:profile", "offline"],
        "scope_sep": " ",
        "pkce": False,
        "token_auth": "body",
        "rotates_refresh_token": True,
        "min_state_len": 8,  # WHOOP requires state >= 8 chars
        "metrics": ["resting_hr", "hrv_ms", "spo2", "sleep_min", "resp_rate"],
        "client_id_env": "WHOOP_CLIENT_ID",
        "client_secret_env": "WHOOP_CLIENT_SECRET",
        "signup_url": "https://developer-dashboard.whoop.com",
        "approval": "instant",
    },
    # ── Garmin ───────────────────────────────────────────────────────────
    # Garmin migrated to OAuth2 + PKCE. Access tokens last ~3 months. Note the
    # developer program has a manual review queue: budget weeks, not minutes.
    "garmin": {
        "id": "garmin",
        "name": "Garmin",
        "tier": "cloud",
        "authorize_url": "https://connect.garmin.com/oauth2Confirm",
        "token_url": "https://diauth.garmin.com/di-oauth2-service/oauth/token",
        "api_base": "https://apis.garmin.com/wellness-api/rest",
        "scopes": ["HEALTH_EXPORT"],
        "scope_sep": " ",
        "pkce": True,  # PKCE is required, not optional
        "token_auth": "body",
        "metrics": ["resting_hr", "hrv_ms", "spo2", "steps", "sleep_min", "resp_rate"],
        "client_id_env": "GARMIN_CLIENT_ID",
        "client_secret_env": "GARMIN_CLIENT_SECRET",
        "signup_url": "https://developerportal.garmin.com/developer-programs/health-api",
        "approval": "review_queue",
    },
    # ── Google Health API ────────────────────────────────────────────────
    # This is the *cloud* successor to the Fitbit Web API (which sunsets
    # 2026-09-30) and covers Fitbit trackers + Pixel Watch. It is NOT Health
    # Connect — Health Connect is on-device Android and has no web API.
    # Tokens do not carry over from Fitbit: every user must re-consent.
    "google_health": {
        "id": "google_health",
        "name": "Google Health",
        "tier": "cloud",
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "api_base": "https://health.googleapis.com/v4",
        "scopes": [
            "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
            "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
            "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
        ],
        "scope_sep": " ",
        "pkce": True,
        "token_auth": "body",
        "extra_authorize_params": {"access_type": "offline", "prompt": "consent"},
        "metrics": ["resting_hr", "hrv_ms", "spo2", "steps", "sleep_min", "resp_rate"],
        "client_id_env": "GOOGLE_HEALTH_CLIENT_ID",
        "client_secret_env": "GOOGLE_HEALTH_CLIENT_SECRET",
        "signup_url": "https://developers.google.com/health/setup",
        "approval": "app_verification",
        "covers": "Fitbit trackers and Pixel Watch",
    },
    # ── Native-only platforms ────────────────────────────────────────────
    "apple_health": {
        "id": "apple_health",
        "name": "Apple Health",
        "tier": "native",
        "metrics": ["resting_hr", "hrv_ms", "spo2", "steps", "sleep_min", "resp_rate"],
        "import_supported": True,
        "covers": "Apple Watch and iPhone",
        "why_gated": (
            "HealthKit is a native iOS framework with no web API. A website can only "
            "read a Health export the user creates manually."
        ),
    },
    "health_connect": {
        "id": "health_connect",
        "name": "Health Connect",
        "tier": "native",
        "metrics": ["resting_hr", "hrv_ms", "spo2", "steps", "sleep_min"],
        "import_supported": False,
        "covers": "Samsung Health, Xiaomi, Noise, boAt and most Android bands",
        "why_gated": (
            "Health Connect is an on-device Android datastore. It shares data only "
            "with apps installed on the phone, never with a website."
        ),
    },
    # ── Browser-local ────────────────────────────────────────────────────
    "ble_band": {
        "id": "ble_band",
        "name": "Bluetooth band or strap",
        "tier": "local",
        "metrics": ["resting_hr", "hrv_ms"],
        "covers": "Any device exposing the standard BLE Heart Rate Service",
    },
    "manual": {
        "id": "manual",
        "name": "Enter it myself",
        "tier": "local",
        "metrics": ["resting_hr", "hrv_ms", "spo2"],
    },
}

CLOUD_PROVIDERS = [p for p, c in PROVIDERS.items() if c["tier"] == "cloud"]


def get_provider(pid: str) -> Dict[str, Any]:
    cfg = PROVIDERS.get(pid)
    if not cfg:
        raise KeyError(f"unknown provider: {pid}")
    return cfg


def credentials(pid: str) -> Dict[str, Optional[str]]:
    """Read this provider's client credentials from the environment."""
    cfg = get_provider(pid)
    if cfg["tier"] != "cloud":
        return {"client_id": None, "client_secret": None}
    return {
        "client_id": os.environ.get(cfg["client_id_env"]) or None,
        "client_secret": os.environ.get(cfg["client_secret_env"]) or None,
    }


def is_configured(pid: str) -> bool:
    """True when this provider has real credentials and can actually connect.

    Drives the UI's "setup pending" state — the app should never show a
    Connect button that is guaranteed to fail.
    """
    cfg = PROVIDERS.get(pid)
    if not cfg or cfg["tier"] != "cloud":
        return False
    creds = credentials(pid)
    # PKCE providers can technically work public-client, but both of ours
    # (Garmin, Google) issue a secret, so require both for a clear signal.
    return bool(creds["client_id"] and creds["client_secret"])


def public_catalog() -> List[Dict[str, Any]]:
    """The provider list the frontend renders. Never leaks secrets."""
    out: List[Dict[str, Any]] = []
    for pid, cfg in PROVIDERS.items():
        out.append(
            {
                "id": pid,
                "name": cfg["name"],
                "tier": cfg["tier"],
                "metrics": cfg.get("metrics", []),
                "configured": is_configured(pid),
                "covers": cfg.get("covers"),
                "why_gated": cfg.get("why_gated"),
                "import_supported": cfg.get("import_supported", False),
                "approval": cfg.get("approval"),
            }
        )
    return out
