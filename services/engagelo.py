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
    templates: Dict[str, str]

    @classmethod
    def from_env(cls) -> "EngageloSettings":
        enabled = os.environ.get("WHATSAPP_INTEGRATION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        settings = cls(
            enabled=enabled,
            base_url=os.environ.get("ENGAGELO_BASE_URL", "https://bot.engagelo.com").rstrip("/"),
            api_key=os.environ.get("ENGAGELO_API_KEY", "").strip(),
            phone_number_id=(os.environ.get("WHATSAPP_PHONE_NUMBER_ID") or os.environ.get("ENGAGELO_WHATSAPP_ACCOUNT_ID", "")).strip(),
            default_country_code=os.environ.get("WHATSAPP_DEFAULT_COUNTRY_CODE", "91").strip().lstrip("+"),
            timeout_seconds=float(os.environ.get("ENGAGELO_TIMEOUT_SECONDS", "10")),
            max_retries=max(0, int(os.environ.get("ENGAGELO_MAX_RETRIES", "2"))),
            templates={
                "otp": os.environ.get("ENGAGELO_TEMPLATE_OTP", "").strip(),
                "scan_complete": os.environ.get("ENGAGELO_TEMPLATE_SCAN_COMPLETE", "").strip(),
                "report_ready": os.environ.get("ENGAGELO_TEMPLATE_REPORT_READY", "").strip(),
                "test_update": os.environ.get("ENGAGELO_TEMPLATE_TEST_UPDATE", "").strip(),
                "reminder": os.environ.get("ENGAGELO_TEMPLATE_REMINDER", "").strip(),
            },
        )
        if settings.enabled:
            missing = [name for name, value in {
                "ENGAGELO_API_KEY": settings.api_key,
                "WHATSAPP_PHONE_NUMBER_ID": settings.phone_number_id,
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
        if not self.settings.enabled:
            return {"sent": False, "disabled": True, "provider_message_id": None}
        payload = {
            "apiToken": self.settings.api_key,
            "phone_number_id": self.settings.phone_number_id,
            "phone_number": "".join(ch for ch in phone if ch.isdigit()),
            "message": message,
        }
        endpoint = f"{self.settings.base_url}/api/v1/whatsapp/send"
        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.timeout_seconds, transport=self.transport) as client:
                    response = await client.post(endpoint, data=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise EngageloDeliveryError("Transient provider failure", status_code=response.status_code, safe_code="transient_provider_error")
                if response.status_code >= 400:
                    raise EngageloDeliveryError("Provider rejected the message", status_code=response.status_code, safe_code="provider_rejected")
                data = response.json()
                if str(data.get("status")) != "1":
                    raise EngageloDeliveryError("Provider did not accept the message", status_code=response.status_code, safe_code="provider_rejected")
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
        return await self.send_message(phone, f"Your DilSay verification code is {otp}. It expires in {ttl_minutes} minutes. Do not share this code.", message_type="otp")

    async def send_scan_complete(self, phone: str) -> Dict[str, Any]:
        return await self.send_message(phone, "Your DilSay heart check is complete. Open DilSay to view your results.", message_type="scan_complete")

    async def send_report_ready(self, phone: str, report_url: Optional[str] = None) -> Dict[str, Any]:
        suffix = f" {report_url}" if report_url else " Open DilSay to view it."
        return await self.send_message(phone, "Your DilSay heart report is ready." + suffix, message_type="report_ready")

    async def send_test_update(self, phone: str) -> Dict[str, Any]:
        return await self.send_message(phone, "Your DilSay test update is now available. Open DilSay to view it.", message_type="test_update")
