"""Focused tests for WhatsApp auth, consent, delivery and webhook behavior."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import unittest
from urllib.parse import parse_qs
from collections import defaultdict
from copy import deepcopy
from unittest.mock import patch

import httpx
from fastapi import BackgroundTasks, HTTPException, Response
from starlette.requests import Request

import server
from services.engagelo import EngageloClient, EngageloDeliveryError, EngageloSettings


def request(body=None, *, cookie="", headers=None, path="/api/test", method="POST"):
    raw = json.dumps(body or {}).encode()
    values = [(b"content-type", b"application/json")]
    if cookie:
        values.append((b"cookie", f"dilsay_session={cookie}".encode()))
    for key, value in (headers or {}).items():
        values.append((key.lower().encode(), value.encode()))
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request({"type": "http", "http_version": "1.1", "method": method, "scheme": "https", "path": path, "raw_path": path.encode(), "query_string": b"", "headers": values, "client": ("test", 1), "server": ("test", 443)}, receive)


def settings(*, retries=0):
    return EngageloSettings(True, "https://provider.test", "secret-api-key", "sender-id", "91", 1, retries)


class AdminAccessTests(unittest.TestCase):
    def test_admin_routes_are_disabled_without_key_in_production(self):
        with patch.object(server, "APP_ENV", "production"), patch.object(server, "ADMIN_KEY", ""):
            with self.assertRaises(HTTPException) as raised:
                server.require_admin(request())
        self.assertEqual(raised.exception.status_code, 403)

    def test_admin_routes_remain_available_without_key_in_development(self):
        with patch.object(server, "APP_ENV", "development"), patch.object(server, "ADMIN_KEY", ""):
            server.require_admin(request())

    def test_configured_admin_key_is_enforced(self):
        with patch.object(server, "APP_ENV", "production"), patch.object(server, "ADMIN_KEY", "test-key"):
            with self.assertRaises(HTTPException) as raised:
                server.require_admin(request())
            self.assertEqual(raised.exception.status_code, 403)
            server.require_admin(request(headers={"x-admin-key": "test-key"}))


class EngageloClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_otp_uses_exact_template_form_contract(self):
        async def handler(req):
            self.assertEqual(str(req.url), "https://provider.test/api/v1/whatsapp/send/template")
            self.assertEqual(parse_qs((await req.aread()).decode()), {
                "apiToken": ["secret-api-key"], "phone_number_id": ["sender-id"],
                "phone_number": ["919876543210"], "template_id": ["454203"],
                "templateVariable-OTP-1": ["123456"],
            })
            return httpx.Response(200, json={"status": "1"})
        result = await EngageloClient(settings(), httpx.MockTransport(handler)).send_otp("+919876543210", "123456", 5)
        self.assertTrue(result["sent"])

    def test_text_sender_needs_only_api_key_and_phone_number_id(self):
        env = {
            "WHATSAPP_INTEGRATION_ENABLED": "true",
            "ENGAGELO_API_KEY": "test-key",
            "WHATSAPP_PHONE_NUMBER_ID": "test-phone-id",
            "ENGAGELO_WEBHOOK_SECRET": "",
        }
        with patch.dict(os.environ, env):
            configured = EngageloSettings.from_env()
        self.assertTrue(configured.enabled)
        self.assertEqual(configured.phone_number_id, "test-phone-id")

    async def test_success_uses_documented_form_contract(self):
        seen = {}

        async def handler(req):
            seen["body"] = (await req.aread()).decode()
            return httpx.Response(200, json={"status": "1", "wa_message_id": "wamid.1"})

        result = await EngageloClient(settings(), httpx.MockTransport(handler)).send_scan_complete("+91 98765 43210")
        self.assertTrue(result["sent"])
        self.assertEqual(result["provider_message_id"], "wamid.1")
        self.assertIn("phone_number=919876543210", seen["body"])
        self.assertIn("phone_number_id=sender-id", seen["body"])
        self.assertIn("apiToken=secret-api-key", seen["body"])

    async def test_non_retryable_provider_error_is_sanitized(self):
        async def handler(_req):
            return httpx.Response(400, json={"status": "0"})

        with self.assertRaises(EngageloDeliveryError) as raised:
            await EngageloClient(settings(retries=2), httpx.MockTransport(handler)).send_test_update("+919876543210")
        self.assertEqual(raised.exception.safe_code, "provider_rejected")

    async def test_closed_conversation_window_has_safe_error_code(self):
        async def handler(_req):
            return httpx.Response(200, json={"status": "0", "message": "Sending message outside 24 hour window is not allowed. You can only send template message to this user."})

        with self.assertRaises(EngageloDeliveryError) as raised:
            await EngageloClient(settings(), httpx.MockTransport(handler)).send_otp("+919876543210", "123456", 5)
        self.assertEqual(raised.exception.safe_code, "outside_24_hour_window")


class WhatsAppFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = defaultdict(dict)
        self.sent = []

        async def put(col, row, doc_id=None):
            self.db[col][doc_id or row["id"]] = deepcopy(row)

        async def get(col, doc_id):
            row = self.db[col].get(doc_id)
            return deepcopy(row) if row else None

        async def rows(col, where=None, limit=None, **_kwargs):
            found = [deepcopy(row) for row in self.db[col].values() if all(row.get(k) == v for k, v in (where or {}).items())]
            return found[:limit] if limit else found

        owner = self

        class Provider:
            def __init__(self, *_args, **_kwargs):
                pass

            async def send_otp(self, phone, otp, ttl):
                owner.sent.append(("otp", phone, otp, ttl))
                return {"sent": True, "provider_message_id": "otp-1"}

            async def send_scan_complete(self, phone):
                owner.sent.append(("scan_complete", phone))
                return {"sent": True, "provider_message_id": "scan-1"}

            async def send_report_ready(self, phone, url=None):
                owner.sent.append(("report_ready", phone, url))
                return {"sent": True, "provider_message_id": "report-1"}

            async def send_test_update(self, phone):
                owner.sent.append(("test_update", phone))
                return {"sent": True, "provider_message_id": "test-1"}

        self.patches = [
            patch.object(server, "fs_put", put),
            patch.object(server, "fs_get", get),
            patch.object(server, "fs_list", rows),
            patch.object(server, "EngageloClient", Provider),
            patch.object(server, "WHATSAPP_SETTINGS", settings()),
            patch.object(server, "WHATSAPP_AUTH_ENABLED", True),
            patch.object(server, "WHATSAPP_DEV_OTP", ""),
            patch.object(server, "OTP_HASH_SECRET", "unit-test-secret"),
            patch.object(server, "AUTH_COOKIE_SECURE", False),
            patch.object(server, "ENGAGELO_WEBHOOK_SECRET", "webhook-test-secret"),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()

    async def authenticate(self, phone="9876543210"):
        await server.request_whatsapp_otp(request({"phone": phone}))
        otp = self.sent[-1][2]
        response = Response()
        result = await server.verify_whatsapp_otp(request({"phone": phone, "otp": otp}), response)
        token = response.headers["set-cookie"].split("dilsay_session=", 1)[1].split(";", 1)[0]
        return result["row"], token

    async def test_fixed_otp_creates_normal_session_without_provider_in_production(self):
        with patch.object(server, "APP_ENV", "production"), patch.object(server, "WHATSAPP_DEV_OTP", "555666"), patch.object(server, "WHATSAPP_SETTINGS", EngageloSettings(False, "https://provider.test", "", "", "91", 1, 0)):
            result = await server.request_whatsapp_otp(request({"phone": "9876543210"}))
            self.assertTrue(result["success"])
            self.assertEqual(result["message"], "OTP ready")
            self.assertEqual(self.sent, [])
            response = Response()
            verified = await server.verify_whatsapp_otp(request({"phone": "9876543210", "otp": "555666"}), response)
        self.assertTrue(verified["row"]["phone_verified_at"])
        self.assertIn("dilsay_session=", response.headers["set-cookie"])

    async def test_resend_is_fresh_and_enabled_provider_ignores_fixed_code(self):
        with patch.object(server, "WHATSAPP_DEV_OTP", "555666"), patch.object(server.secrets, "randbelow", side_effect=[123456, 123456, 654321]):
            await server.request_whatsapp_otp(request({"phone_number": "9876543210"}))
            challenge = next(iter(self.db[server.COL_OTP_CHALLENGES].values()))
            challenge["sent_at_epoch"] -= server.OTP_RESEND_COOLDOWN_SECONDS + 1
            await server.request_whatsapp_otp(request({"phone": "9876543210"}))
        self.assertEqual([row[2] for row in self.sent], ["123456", "654321"])
        with self.assertRaises(HTTPException):
            await server.verify_whatsapp_otp(request({"phone": "9876543210", "otp": "123456"}), Response())

    async def test_phone_normalization_and_otp_success_replay_and_cooldown(self):
        self.assertEqual(server._normalize_phone("98765 43210"), "+919876543210")
        await server.request_whatsapp_otp(request({"phone": "+91 98765 43210"}))
        with self.assertRaises(HTTPException) as cooldown:
            await server.request_whatsapp_otp(request({"phone": "9876543210"}))
        self.assertEqual(cooldown.exception.status_code, 429)
        otp = self.sent[-1][2]
        response = Response()
        verified = await server.verify_whatsapp_otp(request({"phone": "9876543210", "otp": otp}), response)
        self.assertTrue(verified["row"]["phone_verified_at"])
        with self.assertRaises(HTTPException) as replay:
            await server.verify_whatsapp_otp(request({"phone": "9876543210", "otp": otp}), Response())
        self.assertEqual(replay.exception.status_code, 400)

    async def test_wrong_expired_and_max_attempt_otp(self):
        await server.request_whatsapp_otp(request({"phone": "9876543211"}))
        with self.assertRaises(HTTPException) as wrong:
            await server.verify_whatsapp_otp(request({"phone": "9876543211", "otp": "000000"}), Response())
        self.assertEqual(wrong.exception.status_code, 400)
        challenge_id = hashlib.sha256("+919876543211".encode()).hexdigest()
        self.db[server.COL_OTP_CHALLENGES][challenge_id]["attempts"] = server.OTP_MAX_ATTEMPTS
        with self.assertRaises(HTTPException) as limited:
            await server.verify_whatsapp_otp(request({"phone": "9876543211", "otp": self.sent[-1][2]}), Response())
        self.assertEqual(limited.exception.status_code, 429)
        self.db[server.COL_OTP_CHALLENGES][challenge_id]["attempts"] = 0
        self.db[server.COL_OTP_CHALLENGES][challenge_id]["expires_at_epoch"] = time.time() - 1
        with self.assertRaises(HTTPException) as expired:
            await server.verify_whatsapp_otp(request({"phone": "9876543211", "otp": self.sent[-1][2]}), Response())
        self.assertEqual(expired.exception.status_code, 400)

    async def test_consent_grant_revoke_and_no_consent_delivery(self):
        patient, token = await self.authenticate()
        initial = await server.get_whatsapp_consent(request(cookie=token, method="GET"))
        self.assertFalse(initial["consent"])
        await server._deliver_whatsapp_event(patient["id"], "scan_complete", "scan-0")
        self.assertFalse(any(row[0] == "scan_complete" for row in self.sent))
        granted = await server.update_whatsapp_consent(request({"consent": True, "source": "scan_rppg"}, cookie=token, method="PUT"))
        self.assertTrue(granted["consent"])
        revoked = await server.update_whatsapp_consent(request({"consent": False, "source": "profile"}, cookie=token, method="PUT"))
        self.assertFalse(revoked["consent"])
        self.assertEqual(len(self.db[server.COL_WHATSAPP_CONSENTS]), 2)

    async def test_face_finger_and_test_ready_delivery(self):
        _patient, token = await self.authenticate()
        await server.update_whatsapp_consent(request({"consent": True, "source": "scan_rppg"}, cookie=token, method="PUT"))
        for mode in ("face", "finger"):
            tasks = BackgroundTasks()
            result = await server.save_snapshot(request({"mode": mode, "fused": {"bpm": 72}}, cookie=token), tasks)
            await tasks()
            saved = self.db[server.COL_SNAPSHOTS][result["id"]]
            self.assertEqual(saved["mode"], mode)
        tasks = BackgroundTasks()
        await server.save_test_status(request({"testType": "ai_steth", "status": "ready"}, cookie=token), tasks)
        await tasks()
        self.assertEqual([row[0] for row in self.sent].count("scan_complete"), 2)
        self.assertIn("test_update", [row[0] for row in self.sent])

    async def test_provider_failure_does_not_fail_completed_scan(self):
        _patient, token = await self.authenticate()
        await server.update_whatsapp_consent(request({"consent": True, "source": "scan_rppg"}, cookie=token, method="PUT"))

        class FailingProvider:
            def __init__(self, *_args, **_kwargs):
                pass

            async def send_scan_complete(self, _phone):
                raise EngageloDeliveryError("Provider unavailable", safe_code="provider_unavailable")

        tasks = BackgroundTasks()
        with patch.object(server, "EngageloClient", FailingProvider):
            result = await server.save_snapshot(request({"mode": "face", "fused": {"bpm": 70}}, cookie=token), tasks)
            await tasks()
        self.assertTrue(result["ok"])
        delivery = next(iter(self.db[server.COL_WHATSAPP_DELIVERIES].values()))
        self.assertEqual(delivery["status"], "failed")
        self.assertEqual(delivery["error_code"], "provider_unavailable")

    async def test_report_notification_respects_consent(self):
        patient, token = await self.authenticate()

        class Mistral429:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                return httpx.Response(429, text="rate limited")

        risk = {"primary": {"modelUsed": "unit", "score": 1.0, "band": "low"}}
        body = {"inputs": {"age": 40}, "vitals": {"bpm": 72}, "force": True}

        async def generate():
            tasks = BackgroundTasks()
            with patch.object(server, "MISTRAL_KEY", "test"), patch.object(server, "route_and_run", return_value=risk), patch.object(server.httpx, "AsyncClient", return_value=Mistral429()):
                result = await server.health_analyze(request(body, cookie=token), tasks)
                await tasks()
            return result

        without_consent = await generate()
        self.assertTrue(without_consent["reportId"])
        self.assertNotIn("report_ready", [row[0] for row in self.sent])
        await server.update_whatsapp_consent(request({"consent": True, "source": "profile"}, cookie=token, method="PUT"))
        with_consent = await generate()
        self.assertTrue(with_consent["reportId"])
        self.assertIn("report_ready", [row[0] for row in self.sent])
        self.assertEqual(self.db[server.COL_HEALTH_REPORTS][with_consent["reportId"]]["patientId"], patient["id"])

    async def test_webhook_signature_and_duplicate_are_safe(self):
        patient, _token = await self.authenticate()
        self.db[server.COL_WHATSAPP_DELIVERIES]["delivery-1"] = {"id": "delivery-1", "patient_id": patient["id"], "provider_message_id": "wamid.1", "status": "sent"}
        payload = {"event_id": "event-1", "wa_message_id": "wamid.1", "message_status": "delivered"}
        raw = json.dumps(payload).encode()
        signature = hmac.new(b"webhook-test-secret", raw, hashlib.sha256).hexdigest()
        first = await server.engagelo_webhook(request(payload, headers={"x-engagelo-signature": f"sha256={signature}"}, path="/api/webhooks/engagelo"))
        second = await server.engagelo_webhook(request(payload, headers={"x-engagelo-signature": f"sha256={signature}"}, path="/api/webhooks/engagelo"))
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(self.db[server.COL_WHATSAPP_DELIVERIES]["delivery-1"]["status"], "delivered")
        with self.assertRaises(HTTPException) as invalid:
            await server.engagelo_webhook(request(payload, headers={"x-engagelo-signature": "bad"}, path="/api/webhooks/engagelo"))
        self.assertEqual(invalid.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
