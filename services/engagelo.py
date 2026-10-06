"""Server-side Engagelo/WhatsApp client.

No provider credential or full health payload leaves this module through logs.
The public Engagelo API accepts form-encoded fields at /api/v1/whatsapp/send.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


logger = logging.getLogger("dilsay.engagelo")


class EngageloConfigurationError(RuntimeError):
    pass


class EngageloDeliveryError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, safe_code: str = "provider_error"):
        super().__init__(message)
        self.status_code = status_code
        self.safe_code = safe_code


@dataclass(frozen=True)
class EngageloSettings:
    enabled: bool
    base_url: str
    api_key: str
    phone_number_id: str
    default_country_code: str
    timeout_seconds: float
    max_retries: int
    otp_template_id: str = "454203"

    @classmethod
    def from_env(cls) -> "EngageloSettings":
        enabled = os.environ.get("WHATSAPP_INTEGRATION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        settings = cls(
            enabled=enabled,
            base_url=os.environ.get("ENGAGELO_BASE_URL", "https://bot.engagelo.com").rstrip("/"),
            api_key=os.environ.get("ENGAGELO_API_KEY", "").strip(),
            phone_number_id=os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "").strip(),
            default_country_code=os.environ.get("WHATSAPP_DEFAULT_COUNTRY_CODE", "91").strip().lstrip("+"),
            timeout_seconds=float(os.environ.get("ENGAGELO_TIMEOUT_SECONDS", "10")),
            max_retries=max(0, int(os.environ.get("ENGAGELO_MAX_RETRIES", "2"))),
            otp_template_id=os.environ.get("ENGAGELO_OTP_TEMPLATE_ID", "454203").strip(),
        )
        if settings.enabled:
            missing = [name for name, value in {
                "ENGAGELO_API_KEY": settings.api_key,
                "WHATSAPP_PHONE_NUMBER_ID": settings.phone_number_id,
                "ENGAGELO_OTP_TEMPLATE_ID": settings.otp_template_id,
            }.items() if not value]
            if missing:
                raise EngageloConfigurationError("Missing required WhatsApp configuration: " + ", ".join(missing))
        return settings


def mask_phone(phone: str) -> str:
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if len(digits) <= 4:
        return "****"
    prefix = "+" + digits[:2] if len(digits) > 10 else ""
    return f"{prefix}{'*' * max(4, len(digits) - len(prefix.lstrip('+')) - 4)}{digits[-4:]}"


class EngageloClient:
    def __init__(self, settings: Optional[EngageloSettings] = None, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.settings = settings or EngageloSettings.from_env()
        self.transport = transport

    async def send_message(self, phone: str, message: str, *, message_type: str) -> Dict[str, Any]:
        return await self._send(phone, {"message": message}, message_type=message_type, path="/api/v1/whatsapp/send")

    async def _send(self, phone: str, fields: Dict[str, str], *, message_type: str, path: str) -> Dict[str, Any]:
        if not self.settings.enabled:
            return {"sent": False, "disabled": True, "provider_message_id": None}
        payload = {
            "apiToken": self.settings.api_key,
            "phone_number_id": self.settings.phone_number_id,
            "phone_number": "".join(ch for ch in phone if ch.isdigit()),
            **fields,
        }
        endpoint = f"{self.settings.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.timeout_seconds, transport=self.transport, trust_env=False) as client:
                    response = await client.post(endpoint, data=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise EngageloDeliveryError("Transient provider failure", status_code=response.status_code, safe_code="transient_provider_error")
                if response.status_code >= 400:
                    raise EngageloDeliveryError("Provider rejected the message", status_code=response.status_code, safe_code="provider_rejected")
                try:
                    data = response.json()
                except ValueError as exc:
                    raise EngageloDeliveryError("Invalid provider response", safe_code="provider_rejected") from exc
                if not isinstance(data, dict):
                    raise EngageloDeliveryError("Invalid provider response", safe_code="provider_rejected")
                if str(data.get("status")) != "1":
                    provider_message = str(data.get("message") or "").lower()
                    safe_code = "outside_24_hour_window" if "outside 24 hour window" in provider_message else "provider_rejected"
                    raise EngageloDeliveryError("Provider did not accept the message", status_code=response.status_code, safe_code=safe_code)
                provider_id = data.get("wa_message_id") or data.get("message_id")
                logger.info("whatsapp_sent type=%s phone=%s status=%s provider_id=%s retry=%s", message_type, mask_phone(phone), response.status_code, provider_id or "none", attempt)
                return {"sent": True, "disabled": False, "provider_message_id": provider_id, "provider_status": data.get("message")}
            except (httpx.TimeoutException, httpx.NetworkError, EngageloDeliveryError) as exc:
                last_error = exc
                retryable = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)) or getattr(exc, "safe_code", "") == "transient_provider_error"
                logger.warning("whatsapp_send_failed type=%s phone=%s status=%s retry=%s retryable=%s", message_type, mask_phone(phone), getattr(exc, "status_code", None), attempt, retryable)
                if not retryable or attempt >= self.settings.max_retries:
                    break
                await asyncio.sleep(min(2 ** attempt, 4))
        if isinstance(last_error, EngageloDeliveryError):
            raise last_error
        raise EngageloDeliveryError("Unable to reach WhatsApp provider", safe_code="provider_unavailable") from last_error

    async def send_otp(self, phone: str, otp: str, ttl_minutes: int) -> Dict[str, Any]:
        return await self._send(phone, {
            "template_id": self.settings.otp_template_id,
            "templateVariable-OTP-1": otp,
        }, message_type="otp", path="/api/v1/whatsapp/send/template")

    async def send_scan_complete(self, phone: str) -> Dict[str, Any]:
        return await self.send_message(phone, "Your DilSay heart check is complete. Open DilSay to view your results.", message_type="scan_complete")

    async def send_report_ready(self, phone: str, report_url: Optional[str] = None) -> Dict[str, Any]:
        suffix = f" {report_url}" if report_url else " Open DilSay to view it."
        return await self.send_message(phone, "Your DilSay heart report is ready." + suffix, message_type="report_ready")

    async def send_test_update(self, phone: str) -> Dict[str, Any]:
        return await self.send_message(phone, "Your DilSay test update is now available. Open DilSay to view it.", message_type="test_update")
