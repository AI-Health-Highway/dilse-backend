"""
AiSteth Core POC.

Validates the two risky integrations in isolation:
  1) Mistral chat completions with the user-supplied key.
  2) MongoDB round-trip for assessment / snapshot / session_log shapes
     matching the original Node/Express contract.

Run: python scripts/poc_core.py
Exits 0 and prints "POC PASS" on success.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MISTRAL_KEY = os.environ.get("MISTRAL_KEY")
MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME", "test_database")


def fail(msg: str) -> None:
    print(f"POC FAIL: {msg}")
    sys.exit(1)


async def test_mistral() -> None:
    if not MISTRAL_KEY:
        fail("MISTRAL_KEY missing from /app/backend/.env")
    url = "https://api.mistral.ai/v1/chat/completions"
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Reply with one short clinical sentence about a heart rate of 72 bpm "
                    "in a healthy adult. No preamble."
                ),
            }
        ],
        "max_tokens": 60,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {MISTRAL_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if r.status_code != 200:
        fail(f"Mistral HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        fail(f"Mistral parse error: {e} :: {data}")
    if not text:
        fail("Mistral returned empty content")
    print(f"  ✓ Mistral OK ({len(text)} chars): {text[:90]!r}")


async def test_mongo() -> None:
    if not MONGO_URL:
        fail("MONGO_URL missing from /app/backend/.env")
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]

    # Use a dedicated POC sub-namespace to avoid polluting prod collections.
    a_col = db["poc_assessments"]
    s_col = db["poc_snapshots"]
    l_col = db["poc_session_logs"]

    # Clean slate
    await a_col.delete_many({})
    await s_col.delete_many({})
    await l_col.delete_many({})

    now = datetime.now(timezone.utc).isoformat()

    # ── Assessment shape mirroring main.js POST /api/save-assessment ──
    a_id = str(uuid.uuid4())
    assessment = {
        "id": a_id,
        "created_at": now,
        "profile": {
            "age": 45,
            "sex": "male",
            "ethnicity": "white",
            "smoking": "non",
            "townsend": 0,
            "sbp": 128,
            "cholHdlRatio": 4.2,
            "bmi": 24.5,
            "conditions": {"diabetes": False, "af": False, "ckd": False},
            "symptoms": {"chestPain": "no"},
            "lifestyle": {"diet": "med", "exercise": "moderate"},
        },
        "vitals": {"hr": 72, "hrv_ms": 45, "spo2": 98, "brpm": 14, "quality": 0.82},
        "qrisk3Score": 7.3,
        "recommendations": ["Annual cardiovascular check-up", "Continue lifestyle"],
        "aiNarrative": "Sample narrative for POC.",
    }
    await a_col.insert_one(dict(assessment))
    fetched = await a_col.find_one({"id": a_id}, {"_id": 0})
    if not fetched or fetched["profile"]["age"] != 45 or fetched["vitals"]["hr"] != 72:
        fail("Assessment round-trip failed")
    print(f"  ✓ Mongo assessment round-trip id={a_id[:8]}")

    # ── Snapshot shape (heavy samples allowed) ──
    s_id = str(uuid.uuid4())
    snapshot = {
        "id": s_id,
        "created_at": now,
        "assessmentId": a_id,
        "fused": {
            "bpm": 72,
            "hrv_ms": 45,
            "spo2": 98,
            "brpm": 14,
            "quality": 0.82,
            "faceQ": 0.85,
            "fingerQ": 0.0,
        },
        "face": {
            "fs": 30,
            "samples": [0.01 * i for i in range(900)],  # 30 s @ 30 Hz fake
            "result": {"bpm": 72, "hrv_ms": 45, "spo2": 98, "brpm": 14, "quality": 0.85},
        },
        "finger": None,
    }
    await s_col.insert_one(dict(snapshot))
    fetched_full = await s_col.find_one({"id": s_id}, {"_id": 0})
    if not fetched_full or len(fetched_full["face"]["samples"]) != 900:
        fail("Snapshot round-trip failed (samples lost)")

    # Listing query must strip heavy samples — emulate the stripping logic.
    docs = await s_col.find({}, {"_id": 0}).to_list(50)
    stripped = []
    for d in docs:
        face = d.get("face")
        finger = d.get("finger")
        stripped.append(
            {
                "id": d["id"],
                "created_at": d["created_at"],
                "assessmentId": d.get("assessmentId"),
                "fused": d.get("fused"),
                "face": {"result": face.get("result"), "samples": len(face.get("samples") or [])}
                if face
                else None,
                "finger": {"result": finger.get("result"), "samples": len(finger.get("samples") or [])}
                if finger
                else None,
            }
        )
    if not stripped or not isinstance(stripped[0]["face"]["samples"], int):
        fail("Snapshot strip logic failed")
    print(f"  ✓ Mongo snapshot round-trip id={s_id[:8]} (strip OK: {stripped[0]['face']['samples']} samples)")

    # ── Session log shape ──
    l_id = str(uuid.uuid4())
    log = {
        "id": l_id,
        "created_at": now,
        "snapshotId": s_id,
        "assessmentId": a_id,
        "deviceType": "iphone",
        "userAgent": "PocAgent/1.0",
        "lineCount": 3,
        "lines": ["[10:00] start", "[10:01] hr=72", "[10:02] locked"],
    }
    await l_col.insert_one(dict(log))
    fetched_log = await l_col.find_one({"id": l_id}, {"_id": 0})
    if not fetched_log or fetched_log["lineCount"] != 3:
        fail("Log round-trip failed")
    print(f"  ✓ Mongo session-log round-trip id={l_id[:8]}")

    # cleanup
    await a_col.drop()
    await s_col.drop()
    await l_col.drop()
    client.close()


async def main() -> None:
    print("AiSteth POC starting…")
    print(f"  MONGO_URL set: {bool(MONGO_URL)}  MISTRAL_KEY set: {bool(MISTRAL_KEY)}")
    await test_mistral()
    await test_mongo()
    print("POC PASS")


if __name__ == "__main__":
    asyncio.run(main())
