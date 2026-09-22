"""Generic OAuth2 broker shared by every cloud wearable provider.

One implementation, N vendors. The per-vendor differences (PKCE or not, Basic
or body client auth, extra authorize params) are data in `registry.py`, not
branches scattered through the codebase.

Security notes, since this handles health data:
  • `state` is HMAC-signed and carries an expiry, so the callback can verify it
    without server-side session storage. This is the CSRF defence.
  • PKCE verifiers are derived deterministically from the signed state via HMAC,
    so no server-side storage is needed for those either — and the verifier is
    never guessable without the signing secret.
  • The app has no user accounts, so a device proves ownership of its stored
    tokens with a `device_key` it generated. The server only ever stores
    sha256(device_key); a leaked device_id alone cannot pull tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from .registry import credentials, get_provider

STATE_TTL_SECONDS = 15 * 60


def _secret() -> bytes:
    """Signing key for OAuth state. Falls back to a per-process random value.

    A per-process fallback means state issued before a restart stops verifying,
    which fails closed (user retries the connect) rather than open.
    """
    raw = os.environ.get("WEARABLE_STATE_SECRET") or os.environ.get("ADMIN_KEY")
    if not raw:
        global _EPHEMERAL
        try:
            return _EPHEMERAL
        except NameError:
            _EPHEMERAL = secrets.token_bytes(32)
            return _EPHEMERAL
    return raw.encode()


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ── Device identity (stands in for user accounts) ────────────────────────

def device_id_for(device_key: str) -> str:
    """Public, storable id derived from a secret the client keeps."""
    return hashlib.sha256(device_key.encode()).hexdigest()[:32]


def verify_device(device_key: str, device_id: str) -> bool:
    return hmac.compare_digest(device_id_for(device_key), device_id)


# ── Signed state ─────────────────────────────────────────────────────────

def make_state(provider: str, device_id: str, redirect_after: str = "/devices") -> str:
    payload = {
        "p": provider,
        "d": device_id,
        "r": redirect_after,
        "n": _b64u(secrets.token_bytes(9)),
        "t": int(time.time()),
    }
    body = _b64u(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64u(hmac.new(_secret(), body.encode(), hashlib.sha256).digest()[:16])
    return f"{body}.{sig}"


def parse_state(state: str) -> Dict[str, Any]:
    """Verify signature + expiry. Raises ValueError on anything suspicious."""
    try:
        body, sig = state.split(".", 1)
    except ValueError:
        raise ValueError("malformed state")

    expected = _b64u(hmac.new(_secret(), body.encode(), hashlib.sha256).digest()[:16])
    if not hmac.compare_digest(sig, expected):
        raise ValueError("state signature mismatch")

    payload = json.loads(_b64u_decode(body))
    if int(time.time()) - int(payload.get("t", 0)) > STATE_TTL_SECONDS:
        raise ValueError("state expired")
    return payload


# ── PKCE ─────────────────────────────────────────────────────────────────

def pkce_verifier(state: str) -> str:
    """Derive the verifier from the signed state — no server-side storage."""
    return _b64u(hmac.new(_secret(), f"pkce:{state}".encode(), hashlib.sha256).digest())


def pkce_challenge(verifier: str) -> str:
    return _b64u(hashlib.sha256(verifier.encode()).digest())


# ── Authorize URL ────────────────────────────────────────────────────────

def build_authorize_url(provider: str, redirect_uri: str, state: str) -> str:
    from urllib.parse import urlencode

    cfg = get_provider(provider)
    creds = credentials(provider)
    if not creds["client_id"]:
        raise ValueError(f"{provider} is not configured: set {cfg['client_id_env']}")

    params = {
        "response_type": "code",
        "client_id": creds["client_id"],
        "redirect_uri": redirect_uri,
        "scope": cfg["scope_sep"].join(cfg["scopes"]),
        "state": state,
    }
    if cfg.get("pkce"):
        params["code_challenge"] = pkce_challenge(pkce_verifier(state))
        params["code_challenge_method"] = "S256"
    params.update(cfg.get("extra_authorize_params", {}))
    return f"{cfg['authorize_url']}?{urlencode(params)}"


# ── Token exchange / refresh ─────────────────────────────────────────────

def _auth_kwargs(cfg: Dict[str, Any], creds: Dict[str, Any], data: Dict[str, str]) -> Tuple[Dict, Optional[Tuple[str, str]]]:
    """Place client credentials where this vendor expects them."""
    if cfg.get("token_auth") == "basic":
        return data, (creds["client_id"], creds["client_secret"])
    data = dict(data)
    data["client_id"] = creds["client_id"]
    data["client_secret"] = creds["client_secret"]
    return data, None


async def exchange_code(
    provider: str,
    code: str,
    redirect_uri: str,
    state: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    cfg = get_provider(provider)
    creds = credentials(provider)

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    if cfg.get("pkce"):
        data["code_verifier"] = pkce_verifier(state)
    data, auth = _auth_kwargs(cfg, creds, data)

    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        r = await client.post(cfg["token_url"], data=data, auth=auth,
                              headers={"Accept": "application/json"})
        if r.status_code >= 400:
            raise ValueError(f"{provider} token exchange failed ({r.status_code}): {r.text[:300]}")
        return _normalize_token(r.json())
    finally:
        if own:
            await client.aclose()


async def refresh_token(
    provider: str,
    refresh: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    cfg = get_provider(provider)
    creds = credentials(provider)

    data = {"grant_type": "refresh_token", "refresh_token": refresh}
    # WHOOP requires the `offline` scope echoed back on refresh.
    if provider == "whoop":
        data["scope"] = "offline"
    data, auth = _auth_kwargs(cfg, creds, data)

    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        r = await client.post(cfg["token_url"], data=data, auth=auth,
                              headers={"Accept": "application/json"})
        if r.status_code >= 400:
            raise ValueError(f"{provider} refresh failed ({r.status_code}): {r.text[:300]}")
        return _normalize_token(r.json())
    finally:
        if own:
            await client.aclose()


def _normalize_token(raw: Dict[str, Any]) -> Dict[str, Any]:
    expires_in = raw.get("expires_in")
    try:
        expires_in = int(expires_in)
    except (TypeError, ValueError):
        expires_in = 3600
    return {
        "access_token": raw.get("access_token"),
        "refresh_token": raw.get("refresh_token"),
        "expires_at": int(time.time()) + expires_in - 60,  # 60s safety margin
        "scope": raw.get("scope"),
        "token_type": raw.get("token_type", "bearer"),
    }


def is_expired(token: Dict[str, Any]) -> bool:
    return int(token.get("expires_at") or 0) <= int(time.time())
