"""AiSteth FastAPI backend.

Ports the Node/Express API surface 1:1 + Phase-3 extensions:
  - Multi-patient mode (`/api/patients` CRUD + summary + trends)
  - AI narrative cache (hash-keyed Mistral cache: `/api/narrative`)

Persistence: Google Cloud Firestore (Native mode) via the async client.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter
from starlette.middleware.cors import CORSMiddleware

from risk import qrisk3, score2, who_ish, route_and_run, list_models
from echo_centers import find_centers, list_cities, BRAND_META, list_partners
import wearables

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# On Cloud Run, project + credentials come from the runtime service account.
# Locally, use `gcloud auth application-default login` (ADC).
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or None
FIRESTORE_DB = os.environ.get("FIRESTORE_DB", "(default)")
MISTRAL_KEY = os.environ.get("MISTRAL_KEY", "")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
# The app itself is open (public MVP). Only sensitive data endpoints that touch
# the `patients` collection, logs, and stats require this admin key.
# Unset → those endpoints are open too (local dev). Set via deploy.sh.
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

_db: Optional[firestore.AsyncClient] = None


def get_db() -> firestore.AsyncClient:
    """Lazy Firestore client — created on first use so the app imports cleanly
    without credentials (tests, tooling)."""
    global _db
    if _db is None:
        _db = firestore.AsyncClient(project=GCP_PROJECT, database=FIRESTORE_DB)
    return _db

COL_ASSESSMENTS = "assessments"
COL_SNAPSHOTS = "snapshots"
COL_LOGS = "session_logs"
COL_PATIENTS = "patients"
COL_NARRATIVES = "narratives"
COL_NOTIFY = "notify_signups"
COL_HEALTH_REPORTS = "health_reports"
COL_WEARABLE_READINGS = "wearable_readings"
COL_WEARABLE_WAITLIST = "wearable_waitlist"
COL_WEARABLE_TOKENS = "wearable_tokens"


# ── Firestore helpers ─────────────────────────────────────

def _query(col: str, where: Optional[Dict[str, Any]] = None):
    q = get_db().collection(col)
    for field, value in (where or {}).items():
        q = q.where(filter=FieldFilter(field, "==", value))
    return q


async def fs_put(col: str, record: Dict[str, Any], doc_id: Optional[str] = None) -> None:
    await get_db().collection(col).document(doc_id or record["id"]).set(dict(record))


async def fs_get(col: str, doc_id: str) -> Optional[Dict[str, Any]]:
    snap = await get_db().collection(col).document(doc_id).get()
    return snap.to_dict() if snap.exists else None


async def fs_list(
    col: str,
    where: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    order_by: Optional[str] = "created_at",
    descending: bool = True,
) -> List[Dict[str, Any]]:
    q = _query(col, where)
    if order_by:
        q = q.order_by(
            order_by,
            direction=firestore.Query.DESCENDING if descending else firestore.Query.ASCENDING,
        )
    if offset:
        q = q.offset(offset)
    if limit:
        q = q.limit(limit)
    return [s.to_dict() async for s in q.stream()]


async def fs_first(col: str, where: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    rows = await fs_list(col, where=where, limit=1)
    return rows[0] if rows else None


async def fs_count(col: str, where: Optional[Dict[str, Any]] = None) -> int:
    res = await _query(col, where).count().get()
    return int(res[0][0].value)


async def fs_delete_where(col: str, field: str, value: Any) -> int:
    q = _query(col, {field: value})
    n = 0
    batch = get_db().batch()
    async for snap in q.stream():
        batch.delete(snap.reference)
        n += 1
        if n % 400 == 0:
            await batch.commit()
            batch = get_db().batch()
    await batch.commit()
    return n

app = FastAPI(title="AiSteth API")
api = APIRouter(prefix="/api")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("aisteth")


# ── Admin guard ───────────────────────────────────────────
# The app + core scan/report flow are public. These sensitive endpoints
# (anything that reads/exports/deletes patient PII, logs, or aggregate stats)
# require the admin key so a random tester can't harvest or wipe others' data.

def require_admin(request: Request) -> None:
    if not ADMIN_KEY:
        return  # unset → open (local dev)
    supplied = request.headers.get("x-admin-key", "")
    if not secrets.compare_digest(supplied, ADMIN_KEY):
        raise HTTPException(status_code=403, detail="Admin key required")


# ── Helpers ────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if doc is None:
        return None
    doc.pop("_id", None)
    return doc


def _strip_snapshot(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop heavy `samples` arrays for listings."""
    def shrink(side: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not side:
            return None
        samples = side.get("samples")
        return {
            "result": side.get("result"),
            "samples": (len(samples) if isinstance(samples, list) else (samples or 0)),
        }

    return {
        "id": d.get("id"),
        "created_at": d.get("created_at"),
        "assessmentId": d.get("assessmentId"),
        "patientId": d.get("patientId"),
        "fused": d.get("fused"),
        "face": shrink(d.get("face")),
        "finger": shrink(d.get("finger")),
    }


def _narrative_key(payload: Dict[str, Any]) -> str:
    """Stable hash of (profile + vitals + qrisk3Score) for cache lookups."""
    canonical = {
        "profile": payload.get("profile") or {},
        "vitals": payload.get("vitals") or {},
        "qrisk3Score": payload.get("qrisk3Score"),
        "recommendations": payload.get("recommendations") or [],
    }
    raw = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Routes ────────────────────────────────────────────────

@api.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "service": "aisteth", "version": "1.1.0"}


@api.get("/local-ip")
async def local_ip() -> Dict[str, Any]:
    try:
        host = socket.gethostbyname(socket.gethostname())
    except Exception:
        host = None
    return {"ip": host}


@api.get("/config")
async def config() -> Dict[str, Any]:
    # Never return the key itself — only whether AI features are available.
    return {"aiEnabled": bool(MISTRAL_KEY)}


# ── Assessments ───────────────────────────────────────────

@api.post("/save-assessment")
async def save_assessment(request: Request) -> Dict[str, Any]:
    body = await request.json()
    record = {"id": str(uuid.uuid4()), "created_at": now_iso(), **body}
    # Surface patientId at top-level too for indexing convenience.
    if record.get("profile") and record["profile"].get("patientId"):
        record["patientId"] = record["profile"]["patientId"]
    await fs_put(COL_ASSESSMENTS, record)
    p = record.get("profile") or {}
    v = record.get("vitals") or {}
    logger.info(
        "assessment saved id=%s patient=%s qrisk3=%s hr=%s",
        record["id"][:8], record.get("patientId"),
        record.get("qrisk3Score"), v.get("hr"),
    )
    return {"ok": True, "id": record["id"]}


@api.get("/assessments")
async def list_assessments(
    limit: int = Query(100, le=1000),
    offset: int = 0,
    patientId: Optional[str] = None,
) -> Dict[str, Any]:
    where = {"patientId": patientId} if patientId else None
    rows = await fs_list(
        COL_ASSESSMENTS, where=where,
        limit=min(limit, 1000), offset=max(offset, 0),
    )
    total = await fs_count(COL_ASSESSMENTS, where=where)
    return {"ok": True, "total": total, "rows": rows}


@api.get("/assessments/patient/{patient_id}")
async def list_assessments_patient(patient_id: str) -> Dict[str, Any]:
    rows = await fs_list(COL_ASSESSMENTS, where={"patientId": patient_id})
    return {"ok": True, "total": len(rows), "rows": rows}


@api.get("/assessments/{aid}")
async def get_assessment(aid: str) -> Dict[str, Any]:
    doc = await fs_get(COL_ASSESSMENTS, aid)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "row": doc}


# ── Snapshots ─────────────────────────────────────────────

@api.post("/snapshot")
async def save_snapshot(request: Request) -> Dict[str, Any]:
    body = await request.json()
    if not body.get("fused"):
        raise HTTPException(status_code=400, detail="Missing fused result")
    record = {
        "id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "assessmentId": body.get("assessmentId"),
        "patientId": body.get("patientId"),
        "fused": body.get("fused"),
        "face": body.get("face"),
        "finger": body.get("finger"),
    }
    await fs_put(COL_SNAPSHOTS, record)
    fused = record["fused"] or {}
    logger.info(
        "snapshot saved id=%s patient=%s bpm=%s quality=%s",
        record["id"][:8], record.get("patientId"),
        fused.get("bpm"), fused.get("quality"),
    )
    return {"ok": True, "id": record["id"]}


@api.get("/snapshots")
async def list_snapshots(
    limit: int = Query(50, le=500),
    offset: int = 0,
    patientId: Optional[str] = None,
) -> Dict[str, Any]:
    where = {"patientId": patientId} if patientId else None
    docs = await fs_list(
        COL_SNAPSHOTS, where=where,
        limit=min(limit, 500), offset=max(offset, 0),
    )
    rows = [_strip_snapshot(r) for r in docs]
    total = await fs_count(COL_SNAPSHOTS, where=where)
    return {"ok": True, "total": total, "rows": rows}


@api.get("/snapshots/{sid}")
async def get_snapshot(sid: str) -> Dict[str, Any]:
    doc = await fs_get(COL_SNAPSHOTS, sid)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "row": doc}


# ── Logs ──────────────────────────────────────────────────

@api.post("/logs")
async def save_log(request: Request) -> Dict[str, Any]:
    body = await request.json()
    lines = body.get("lines")
    if not isinstance(lines, list) or not lines:
        raise HTTPException(status_code=400, detail="lines must be a non-empty array")
    record = {
        "id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "snapshotId": body.get("snapshotId"),
        "assessmentId": body.get("assessmentId"),
        "deviceType": body.get("deviceType"),
        "userAgent": body.get("userAgent"),
        "lineCount": len(lines),
        "lines": lines,
    }
    await fs_put(COL_LOGS, record)
    return {"ok": True, "id": record["id"]}


@api.get("/logs")
async def list_logs(
    request: Request, limit: int = Query(50, le=500), offset: int = 0
) -> Dict[str, Any]:
    require_admin(request)
    rows = await fs_list(COL_LOGS, limit=min(limit, 500), offset=max(offset, 0))
    for r in rows:
        r.pop("lines", None)  # keep listings light
    total = await fs_count(COL_LOGS)
    return {"ok": True, "total": total, "rows": rows}


@api.get("/logs/{lid}")
async def get_log(lid: str, request: Request) -> Dict[str, Any]:
    require_admin(request)
    doc = await fs_get(COL_LOGS, lid)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "row": doc}


# ── Stats ─────────────────────────────────────────────────

@api.get("/stats")
async def stats(request: Request) -> Dict[str, Any]:
    require_admin(request)
    total = await fs_count(COL_SNAPSHOTS)
    total_assessments = await fs_count(COL_ASSESSMENTS)
    total_patients = await fs_count(COL_PATIENTS)
    latest_doc = await fs_first(COL_SNAPSHOTS)
    fused = (latest_doc or {}).get("fused") or {}
    return {
        "ok": True,
        "totalScans": total,
        "totalAssessments": total_assessments,
        "totalPatients": total_patients,
        "latestBpm": fused.get("bpm"),
        "latestHrv": fused.get("hrv_ms"),
        "latestSpo2": fused.get("spo2"),
        "latestBrpm": fused.get("brpm"),
        "latestAt": (latest_doc or {}).get("created_at"),
    }


# ── Patients (multi-patient mode) ─────────────────────────

def _normalize_phone(raw: str) -> str:
    """Keep digits (and a leading +). Enough to dedupe on."""
    raw = (raw or "").strip()
    plus = raw.startswith("+")
    digits = "".join(ch for ch in raw if ch.isdigit())
    return ("+" + digits) if plus else digits


@api.post("/patients")
async def create_patient(request: Request) -> Dict[str, Any]:
    """Create-or-get a patient keyed by phone number.

    Phone is the only required field (sign-in replacement). Sex/ethnicity are
    optional and can be filled later via the health profile. Idempotent: the same
    phone always maps to the same patient record, so onboarding never duplicates.
    """
    body = await request.json()
    phone = _normalize_phone(body.get("phone") or "")
    if len(phone.lstrip("+")) < 7:
        raise HTTPException(status_code=400, detail="A valid phone number is required")

    # Reject obvious identifiers leaking in (privacy-by-design)
    forbidden = {"name", "fullName", "firstName", "lastName", "email"}
    if any(k in body and body[k] for k in forbidden):
        raise HTTPException(
            status_code=400,
            detail="AiSteth does not accept names or email addresses (GDPR/HIPAA data-minimisation).",
        )

    # Idempotent lookup — return the existing patient for this phone if present.
    # No order_by → single-field filter, so no composite index required.
    _matches = await fs_list(COL_PATIENTS, where={"phone": phone}, limit=1, order_by=None)
    existing = _matches[0] if _matches else None
    if existing:
        # Backfill optional fields if the caller now supplies them.
        patch = {k: body[k] for k in ("sex", "ethnicity", "dob", "notes")
                 if body.get(k) and not existing.get(k)}
        if patch:
            existing.update(patch)
            await fs_put(COL_PATIENTS, existing)
        return {"ok": True, "row": existing, "existing": True}

    code = _make_patient_code()
    record = {
        "id": str(uuid.uuid4()),
        "code": code,
        "created_at": now_iso(),
        "phone": phone,
        "sex": (body.get("sex") or "").strip() or None,
        "ethnicity": (body.get("ethnicity") or "").strip() or None,
        "dob": body.get("dob"),
        "notes": body.get("notes"),
        "color": body.get("color") or _pick_color(code),
    }
    await fs_put(COL_PATIENTS, record)
    logger.info("patient created id=%s code=%s", record["id"][:8], code)
    return {"ok": True, "row": record, "existing": False}


def _make_patient_code() -> str:
    """Pseudonymous identifier — no PII embedded. e.g. 'PT-9C4F2A'."""
    import secrets
    return "PT-" + secrets.token_hex(3).upper()


def _pick_color(seed: str) -> str:
    palette = ["#22D3A4", "#10B981", "#34D399", "#06B6D4", "#0EA5E9",
               "#F59E0B", "#F97316", "#EC4899", "#A855F7", "#8B5CF6"]
    h = sum(ord(c) for c in seed)
    return palette[h % len(palette)]


@api.get("/patients")
async def list_patients(request: Request) -> Dict[str, Any]:
    require_admin(request)
    rows = await fs_list(COL_PATIENTS)
    # Attach lightweight aggregates per patient (scanCount + last vitals).
    for r in rows:
        pid = r["id"]
        scan_count = await fs_count(COL_SNAPSHOTS, where={"patientId": pid})
        assessment_count = await fs_count(COL_ASSESSMENTS, where={"patientId": pid})
        latest = await fs_first(COL_SNAPSHOTS, where={"patientId": pid})
        r["scanCount"] = scan_count
        r["assessmentCount"] = assessment_count
        r["latestBpm"] = (latest or {}).get("fused", {}).get("bpm")
        r["latestAt"] = (latest or {}).get("created_at")
    return {"ok": True, "total": len(rows), "rows": rows}


@api.get("/patients/{pid}")
async def get_patient(pid: str, request: Request) -> Dict[str, Any]:
    require_admin(request)
    doc = await fs_get(COL_PATIENTS, pid)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    # Build vitals trend (latest 50 snapshots) for the patient.
    snap_docs = await fs_list(COL_SNAPSHOTS, where={"patientId": pid}, limit=50)
    snaps = [_strip_snapshot(s) for s in snap_docs]
    # Latest assessment
    last_assessment = await fs_first(COL_ASSESSMENTS, where={"patientId": pid})
    trend = [
        {
            "created_at": s.get("created_at"),
            "bpm": (s.get("fused") or {}).get("bpm"),
            "hrv": (s.get("fused") or {}).get("hrv_ms"),
            "spo2": (s.get("fused") or {}).get("spo2"),
            "brpm": (s.get("fused") or {}).get("brpm"),
            "quality": (s.get("fused") or {}).get("quality"),
        }
        for s in snaps
    ]
    return {
        "ok": True,
        "row": doc,
        "trend": list(reversed(trend)),  # oldest → newest for charts
        "scanCount": len(snaps),
        "latestAssessment": last_assessment,
    }


@api.delete("/patients/{pid}")
async def delete_patient(pid: str, request: Request) -> Dict[str, Any]:
    require_admin(request)
    if not await fs_get(COL_PATIENTS, pid):
        raise HTTPException(status_code=404, detail="Not found")
    await get_db().collection(COL_PATIENTS).document(pid).delete()
    return {"ok": True}


# ── GDPR / HIPAA data-subject endpoints ───────────────────

@api.get("/patients/{pid}/export")
async def export_patient(pid: str, request: Request) -> Dict[str, Any]:
    """GDPR Article 20 — data portability. Returns ALL data linked to the patient."""
    require_admin(request)
    p = await fs_get(COL_PATIENTS, pid)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    snaps = await fs_list(COL_SNAPSHOTS, where={"patientId": pid}, limit=10_000)
    assess = await fs_list(COL_ASSESSMENTS, where={"patientId": pid}, limit=10_000)
    return {
        "ok": True,
        "exported_at": now_iso(),
        "patient": p,
        "snapshots": snaps,
        "assessments": assess,
        "notice": "This export contains all personal data AiSteth holds for this pseudonymous patient code.",
    }


@api.post("/patients/{pid}/forget")
async def forget_patient(pid: str, request: Request) -> Dict[str, Any]:
    """GDPR Article 17 — right to erasure. Cascade-deletes everything for this patient."""
    require_admin(request)
    p = await fs_get(COL_PATIENTS, pid)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    n_snaps = await fs_delete_where(COL_SNAPSHOTS, "patientId", pid)
    n_assess = await fs_delete_where(COL_ASSESSMENTS, "patientId", pid)
    await get_db().collection(COL_PATIENTS).document(pid).delete()
    logger.info("patient erased pid=%s snaps=%s asses=%s", pid[:8], n_snaps, n_assess)
    return {
        "ok": True,
        "deleted": {
            "patient": 1,
            "snapshots": n_snaps,
            "assessments": n_assess,
        },
    }


@api.get("/privacy")
async def privacy_notice() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": "1.0",
        "principles": [
            "Data minimisation — Somatic does not collect names or email addresses.",
            "Pseudonymisation — every patient is referenced by an opaque code (PT-XXXXXX).",
            "Local processing — the camera stream is processed in the browser; only computed vitals leave the device.",
            "Right to access — GET /api/patients/{id}/export returns all linked data.",
            "Right to erasure — POST /api/patients/{id}/forget cascade-deletes everything tied to a patient.",
            "Encryption in transit — all traffic is TLS via the Emergent ingress.",
        ],
        "controller": "Somatic (self-hosted deployment)",
        "contact": "your-deployment-admin@example.org",
    }


# ── Email capture (locked-feature interest) ────────────────


@api.post("/notify-me")
async def notify_me(request: Request) -> Dict[str, Any]:
    body = await request.json()
    feature = (body.get("feature") or "").strip()
    email = (body.get("email") or "").strip()
    if not feature or not email or "@" not in email:
        raise HTTPException(status_code=400, detail="feature and a valid email are required")
    record = {
        "id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "feature": feature,
        "email": email,
    }
    await fs_put(COL_NOTIFY, record)
    return {"ok": True}


# ── Narrative cache (Mistral) ─────────────────────────────

@api.post("/narrative")
async def generate_narrative(request: Request) -> Dict[str, Any]:
    """Returns a Mistral-generated clinical narrative for the given context.

    Caches by hash of (profile + vitals + qrisk3Score + recommendations).
    Body: { profile, vitals, qrisk3Score, recommendations, force?: bool }
    """
    body = await request.json()
    if not MISTRAL_KEY:
        raise HTTPException(status_code=503, detail="MISTRAL_KEY not configured")

    key = _narrative_key(body)
    force = bool(body.get("force"))

    if not force:
        cached = await fs_get(COL_NARRATIVES, key)
        if cached:
            logger.info("narrative cache HIT key=%s…", key[:10])
            return {
                "ok": True,
                "text": cached["text"],
                "cached": True,
                "key": key,
                "created_at": cached.get("created_at"),
            }

    profile = body.get("profile") or {}
    vitals = body.get("vitals") or {}
    qrisk = body.get("qrisk3Score")
    recs = body.get("recommendations") or []

    prompt = _build_prompt(profile, vitals, qrisk, recs)
    try:
        async with httpx.AsyncClient(timeout=45) as cli:
            r = await cli.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 420,
                    "temperature": 0.4,
                },
            )
        if r.status_code != 200:
            raise HTTPException(
                status_code=502, detail=f"Mistral HTTP {r.status_code}"
            )
        text = r.json()["choices"][0]["message"]["content"].strip()
    except HTTPException:
        raise
    except Exception as e:  # network/etc.
        raise HTTPException(status_code=502, detail=f"Mistral error: {e}")

    record = {
        "id": str(uuid.uuid4()),
        "key": key,
        "created_at": now_iso(),
        "profile": profile,
        "vitals": vitals,
        "qrisk3Score": qrisk,
        "recommendations": recs,
        "text": text,
    }
    await fs_put(COL_NARRATIVES, record, doc_id=key)
    logger.info("narrative cache MISS key=%s… stored", key[:10])
    return {
        "ok": True, "text": text, "cached": False,
        "key": key, "created_at": record["created_at"],
    }


def _build_prompt(
    profile: Dict[str, Any], vitals: Dict[str, Any],
    qrisk: Optional[float], recs: List[Any],
) -> str:
    rec_lines = "\n".join(
        f"- [{r.get('priority', '').upper() if isinstance(r, dict) else ''}] "
        f"{r.get('test', '') if isinstance(r, dict) else r}"
        for r in (recs or [])
    ) or "(none)"
    return (
        "You are a careful clinical decision-support assistant. Produce a calm, "
        "non-alarming 4-6 sentence narrative aimed at a patient (not a clinician), "
        "summarising the cardiovascular picture from the data below. Include: one "
        "sentence on the vitals, one on the risk score and what it means in plain "
        "language, one on the top 1-2 most important next steps, and a closing "
        "reassurance/encouragement. Do not invent values. End with: "
        "'This is not a diagnosis.'\n\n"
        f"Vitals: HR={vitals.get('hr')} bpm, HRV={vitals.get('hrv_ms')} ms, "
        f"SpO2={vitals.get('spo2')}%, RR={vitals.get('brpm')}/min.\n"
        f"Profile: age={profile.get('age')}, sex={profile.get('sex')}, "
        f"BMI={profile.get('bmi')}, SBP={profile.get('sbp')}, "
        f"Chol/HDL={profile.get('cholHdl')}, smoking={profile.get('smoking')}, "
        f"ethnicity={profile.get('ethnicity')}.\n"
        f"QRISK3 (10-yr CVD risk): {qrisk}%.\n"
        f"Recommendations:\n{rec_lines}"
    )


# ── Health Insights (hybrid, opt-in) ──────────────────────
# Risk scoring + Mistral structured analysis + Louise-Hay-inspired affirmations
# linked to conditions. Strictly framed as educational, NOT a diagnosis.


@api.get("/health/models")
async def health_models() -> Dict[str, Any]:
    return {"ok": True, "models": list_models()}


@api.post("/health/risk")
async def health_risk(request: Request) -> Dict[str, Any]:
    """Compute CV risk using QRISK3/SCORE2/WHO-ISH (router decides primary).

    Body: {
      region: 'uk'|'europe'|'india'|'south_asia'|'global',
      ethnicity: '...', age, sex, sbp, bmi, chol_hdl_ratio, total_cholesterol,
      hdl, townsend, smoking, diabetes_type, family_history (bool),
      treated_hypertension (bool), ckd, af, ra, migraine, sle,
      severe_mental_illness, corticosteroids, atypical_antipsychotics,
      erectile_dysfunction, score2_region, who_subregion
    }
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid body")
    result = route_and_run(body)
    return {"ok": True, "result": result}


def _build_health_analysis_prompt(
    inputs: Dict[str, Any],
    risk_result: Dict[str, Any],
    vitals: Optional[Dict[str, Any]],
) -> str:
    primary = risk_result.get("primary") or {}
    secondary = risk_result.get("secondary") or []
    sec_lines = "\n".join(
        f"  - {s.get('modelUsed')}: {s.get('score')}% ({s.get('band')})"
        for s in secondary
    ) or "  (none)"
    vit = vitals or {}
    return (
        "You are a calm, careful health-education assistant. Produce STRICT JSON only — "
        "no preamble, no markdown fences. The user has just measured their heart "
        "and provided self-reported inputs for cardiovascular risk estimation.\n\n"
        "TONE RULES:\n"
        "- Do NOT diagnose. Do NOT prescribe.\n"
        "- Use plain language a non-clinician can understand.\n"
        "- Be non-alarming and respectful of self-reported uncertainty.\n"
        "- Tests/lifestyle are SUGGESTIONS to discuss with a clinician.\n"
        "- Do NOT include affirmations, spiritual content, or belief-pattern language.\n\n"
        "OUTPUT JSON SHAPE:\n"
        "{\n"
        '  "summary": "<2-3 sentence plain-language risk explanation referencing the model used>",\n'
        '  "tests": [ {"name":"...", "why":"...", "priority":"routine|soon|urgent"} ],   // exactly 3\n'
        '  "lifestyle": [ {"action":"...", "why":"...", "effortLevel":"low|moderate|high"} ], // exactly 3\n'
        '  "disclaimer": "Not a diagnosis. Consider speaking with a clinician."\n'
        "}\n\n"
        f"Primary model: {primary.get('modelUsed')} → {primary.get('score')}% (band: {primary.get('band')}).\n"
        f"Secondary models for comparison:\n{sec_lines}\n"
        f"Inputs used: {primary.get('inputsUsed')}\n"
        f"Missing inputs: {primary.get('missingInputs')}\n"
        f"Vitals (from recent scan): HR={vit.get('bpm')} bpm, HRV={vit.get('hrv_ms')} ms.\n"
        f"User region: {inputs.get('region')}, ethnicity: {inputs.get('ethnicity')}.\n"
        "Return ONLY the JSON."
    )


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first {...} JSON object from a string."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Heuristic extraction
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start: end + 1])
    except Exception:
        return None


def _fallback_health_analysis(risk_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a conservative educational report when Mistral is rate-limited."""
    primary = risk_result.get("primary") or {}
    model = primary.get("modelUsed") or "cardiovascular risk model"
    score = primary.get("score")
    band = str(primary.get("band") or "estimated").lower()
    try:
        score_text = f"{float(score):.1f}%"
    except (TypeError, ValueError):
        score_text = "the available estimate"
    priority = "soon" if band in {"high", "very high"} else "routine"
    return {
        "summary": (
            f"The {model} calculation estimates cardiovascular risk at {score_text}, "
            f"in the {band} band. This is an educational estimate based on the "
            "information provided and should be interpreted with a clinician."
        ),
        "tests": [
            {
                "name": "Blood-pressure review",
                "why": "Repeat measurements can confirm whether the entered reading reflects your usual blood pressure.",
                "priority": priority,
            },
            {
                "name": "Lipid profile",
                "why": "Cholesterol results help a clinician assess cardiovascular risk more completely.",
                "priority": priority,
            },
            {
                "name": "Blood-glucose screening",
                "why": "Glucose or HbA1c testing can identify another important cardiovascular risk factor.",
                "priority": priority,
            },
        ],
        "lifestyle": [
            {
                "action": "Build up regular physical activity",
                "why": "Consistent activity supports cardiovascular fitness; choose an amount appropriate for your health.",
                "effortLevel": "moderate",
            },
            {
                "action": "Choose a heart-supportive eating pattern",
                "why": "More vegetables, fruit, whole grains and less excess salt can support blood pressure and cholesterol.",
                "effortLevel": "moderate",
            },
            {
                "action": "Review tobacco, sleep and stress habits",
                "why": "These habits can materially affect long-term cardiovascular health.",
                "effortLevel": "low",
            },
        ],
        "disclaimer": "Not a diagnosis. Consider speaking with a clinician.",
    }


@api.post("/health/analyze")
async def health_analyze(request: Request) -> Dict[str, Any]:
    """Generate a structured Mistral health analysis (single call).

    Body: { inputs: {...}, vitals?: {bpm,hrv_ms}, snapshotId?, patientId?, force? }
    """
    if not MISTRAL_KEY:
        raise HTTPException(status_code=503, detail="MISTRAL_KEY not configured")
    body = await request.json()
    inputs = body.get("inputs") or {}
    vitals = body.get("vitals") or {}
    snapshot_id = body.get("snapshotId")
    patient_id = body.get("patientId")

    # 1) Compute risk first (deterministic).
    risk_result = route_and_run(inputs)

    # 2) Cache by SHA-256 of (key inputs + primary risk).
    primary = risk_result.get("primary") or {}
    cache_payload = {
        "inputs": {k: inputs.get(k) for k in [
            "region", "ethnicity", "age", "sex", "sbp", "bmi", "chol_hdl_ratio",
            "total_cholesterol", "hdl", "smoking", "diabetes_type",
            "family_history", "treated_hypertension", "ckd", "af",
        ]},
        "vitals": {"bpm_b": round((vitals.get("bpm") or 0) / 5) * 5,
                   "hrv_b": round((vitals.get("hrv_ms") or 0) / 5) * 5},
        "modelUsed": primary.get("modelUsed"),
        "scoreBucket": round((primary.get("score") or 0) / 2) * 2,  # 2% buckets
    }
    key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()

    if not body.get("force"):
        cached = await fs_get(COL_NARRATIVES, f"ha:{key}")
        if cached:
            return {
                "ok": True,
                "cached": True,
                "key": key,
                "risk": risk_result,
                "analysis": cached.get("analysis"),
                "snapshotId": snapshot_id,
            }

    # 3) Mistral call
    prompt = _build_health_analysis_prompt(inputs, risk_result, vitals)
    analysis: Optional[Dict[str, Any]] = None
    try:
        async with httpx.AsyncClient(timeout=60) as cli:
            r = await cli.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 900,
                    "temperature": 0.55,
                    "response_format": {"type": "json_object"},
                },
            )
        if r.status_code == 429:
            logger.warning("Mistral rate limit reached; using local health-analysis fallback")
            analysis = _fallback_health_analysis(risk_result)
        elif r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Mistral HTTP {r.status_code}: {r.text[:200]}")
        else:
            text = r.json()["choices"][0]["message"]["content"].strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mistral error: {e}")

    if analysis is None:
        analysis = _safe_json_extract(text) or {
            "summary": text,
            "tests": [],
            "lifestyle": [],
            "affirmations": [],
            "disclaimer": "Not a diagnosis. Consider speaking with a clinician.",
        }

    record = {
        "id": str(uuid.uuid4()),
        "key": key,
        "kind": "health_analysis",
        "created_at": now_iso(),
        "inputs": inputs,
        "vitals": vitals,
        "risk": risk_result,
        "analysis": analysis,
        "snapshotId": snapshot_id,
        "patientId": patient_id,
    }
    await fs_put(COL_NARRATIVES, record, doc_id=f"ha:{key}")

    # Persist the full report so it can be reopened later from "Recent reports".
    report = {
        "id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "snapshotId": snapshot_id,
        "patientId": patient_id,
        "region": inputs.get("region"),
        "ethnicity": inputs.get("ethnicity"),
        "primaryModel": primary.get("modelUsed"),
        "primaryScore": primary.get("score"),
        "primaryBand": primary.get("band"),
        "summary": analysis.get("summary"),
        # Full payload for the report-detail view:
        "risk": risk_result,
        "analysis": analysis,
        "vitals": vitals,
    }
    await fs_put(COL_HEALTH_REPORTS, report)

    return {
        "ok": True,
        "cached": False,
        "key": key,
        "risk": risk_result,
        "analysis": analysis,
        "reportId": report["id"],
        "snapshotId": snapshot_id,
    }


@api.get("/health/reports")
async def list_health_reports(limit: int = Query(20, le=200)) -> Dict[str, Any]:
    rows = await fs_list(COL_HEALTH_REPORTS, limit=min(limit, 200))
    return {"ok": True, "total": len(rows), "rows": rows}


@api.get("/health/reports/{rid}")
async def get_health_report(rid: str) -> Dict[str, Any]:
    doc = await fs_get(COL_HEALTH_REPORTS, rid)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "row": doc}


# ── Echo booking centers (India seed) ──────────────────────


@api.get("/echo-centers")
async def echo_centers(city: Optional[str] = None, brand: Optional[str] = None, limit: int = Query(50, le=100)) -> Dict[str, Any]:
    """Return the diagnostic-partner brand cards.

    The frontend uses the device's geolocation + `mapsQuery` to open Google Maps
    with a nearby search per brand, so users always get fresh + closest results.
    """
    partners = list_partners()
    if brand:
        partners = [p for p in partners if p["brand"] == brand.lower()]
    return {
        "ok": True,
        "total": len(partners),
        "brands": {p["brand"]: p for p in list_partners()},
        "rows": partners[:max(1, min(limit, 100))],
    }


# ── Wearables / connected devices ──────────────────────────
#
# The client is local-first: readings live in localStorage and the app works
# fully offline. These endpoints are an *optional* mirror, so the day a native
# wrapper or an aggregator lands there is already a server-side shape to write
# into — and so the waitlist gives a real demand signal for building it.

WEARABLE_SOURCES = {"apple_health", "google_health", "ble_band", "manual"}
_WEARABLE_NUMERIC = {
    "resting_hr": (25, 240),
    "hrv_ms": (1, 400),
    "spo2": (50, 100),
    "steps": (0, 200_000),
    "sleep_min": (0, 1440),
    "resp_rate": (4, 60),
}


def _clean_reading(body: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only known metrics inside physiologically plausible bounds."""
    out: Dict[str, Any] = {}
    for key, (lo, hi) in _WEARABLE_NUMERIC.items():
        raw = body.get(key)
        if raw in (None, ""):
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if lo <= val <= hi:
            out[key] = int(round(val))
    return out


@api.post("/wearable/reading")
async def save_wearable_reading(request: Request) -> Dict[str, Any]:
    """Mirror one day's wearable reading. Body: { date, source, resting_hr, ... }"""
    body = await request.json()

    source = (body.get("source") or "manual").strip()
    if source not in WEARABLE_SOURCES:
        raise HTTPException(status_code=400, detail=f"source must be one of {sorted(WEARABLE_SOURCES)}")

    date = (body.get("date") or "").strip() or now_iso()[:10]
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    metrics = _clean_reading(body)
    if not metrics:
        raise HTTPException(status_code=400, detail="no valid metrics in request")

    # Deterministic id => re-posting the same day upserts instead of duplicating.
    device_id = (body.get("device_id") or "local").strip()[:64]
    doc_id = f"{device_id}_{source}_{date}"

    record = {
        "id": doc_id,
        "created_at": now_iso(),
        "date": date,
        "source": source,
        "device_id": device_id,
        **metrics,
    }
    await fs_put(COL_WEARABLE_READINGS, record, doc_id=doc_id)
    return {"ok": True, "row": record}


@api.get("/wearable/readings")
async def list_wearable_readings(
    device_id: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = Query(60, le=365),
) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if device_id:
        where["device_id"] = device_id
    if source:
        where["source"] = source
    rows = await fs_list(COL_WEARABLE_READINGS, where=where or None, limit=limit)
    rows = [_clean(r) for r in rows]
    rows.sort(key=lambda r: r.get("date") or "", reverse=True)
    return {"ok": True, "total": len(rows), "rows": rows}


@api.post("/wearable/waitlist")
async def wearable_waitlist(request: Request) -> Dict[str, Any]:
    """Demand signal for native-only sources (Apple Health / Health Connect).

    Contact is optional on purpose — an anonymous tap still counts as a vote,
    and forcing an email here would suppress the very number we want to read.
    """
    body = await request.json()
    source = (body.get("source") or "").strip()
    if source not in WEARABLE_SOURCES:
        raise HTTPException(status_code=400, detail=f"source must be one of {sorted(WEARABLE_SOURCES)}")

    record = {
        "id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "source": source,
        "contact": (body.get("contact") or "").strip()[:120] or None,
        "platform": (body.get("platform") or "unknown").strip()[:32],
    }
    await fs_put(COL_WEARABLE_WAITLIST, record)
    return {"ok": True}


# ── Wearable providers: OAuth connect / sync ───────────────
#
# One generic broker drives every cloud vendor; the per-vendor differences live
# in `wearables/registry.py`. See WEARABLES.md for the credentials each needs.
#
# Auth model: this app has no user accounts. A device generates a random
# `device_key`, keeps it locally, and the server stores tokens under
# sha256(device_key). Reading or refreshing tokens requires presenting the key,
# so a leaked device_id is not on its own enough to reach someone's health data.

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")


def _redirect_uri(request: Request, provider: str) -> str:
    """The callback URL registered with the vendor. Must match byte-for-byte."""
    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return f"{base}/api/wearable/callback/{provider}"


def _token_doc_id(device_id: str, provider: str) -> str:
    return f"{device_id}_{provider}"


@api.get("/wearable/providers")
async def wearable_providers() -> Dict[str, Any]:
    """Catalog the frontend renders. Reports which vendors have credentials."""
    return {"ok": True, "providers": wearables.public_catalog()}


@api.get("/wearable/connect/{provider}")
async def wearable_connect(provider: str, request: Request, device_id: str = Query(...)) -> Dict[str, Any]:
    """Return the vendor's consent URL. The client opens it at top level."""
    try:
        cfg = wearables.get_provider(provider)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown provider")
    if cfg["tier"] != "cloud":
        raise HTTPException(status_code=400, detail=f"{provider} has no web OAuth flow")
    if not wearables.is_configured(provider):
        raise HTTPException(
            status_code=503,
            detail=f"{provider} is not set up yet: {cfg['client_id_env']} is missing",
        )
    if not device_id or len(device_id) < 16:
        raise HTTPException(status_code=400, detail="a device_id is required")

    state = wearables.make_state(provider, device_id)
    url = wearables.build_authorize_url(provider, _redirect_uri(request, provider), state)
    return {"ok": True, "authorize_url": url, "state": state}


@api.get("/wearable/callback/{provider}")
async def wearable_callback(
    provider: str,
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    """Vendor redirect target. Exchanges the code, stores tokens, bounces home."""
    from starlette.responses import RedirectResponse

    def back(status: str) -> RedirectResponse:
        base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
        return RedirectResponse(f"{base}/devices?connect={provider}&status={status}", status_code=302)

    if error or not code or not state:
        return back("denied" if error == "access_denied" else "error")

    try:
        payload = wearables.parse_state(state)
    except ValueError:
        # Bad or expired state — never proceed with a code we can't attribute.
        return back("expired")
    if payload.get("p") != provider:
        return back("error")

    try:
        token = await wearables.exchange_code(
            provider, code, _redirect_uri(request, provider), state
        )
    except Exception as exc:
        logger.warning("wearable callback %s: %s", provider, exc)
        return back("error")

    device_id = payload["d"]
    await fs_put(
        COL_WEARABLE_TOKENS,
        {
            "id": _token_doc_id(device_id, provider),
            "created_at": now_iso(),
            "device_id": device_id,
            "provider": provider,
            **token,
        },
        doc_id=_token_doc_id(device_id, provider),
    )
    return back("connected")


async def _valid_token(device_key: str, provider: str) -> Dict[str, Any]:
    """Load this device's token, proving ownership and refreshing if stale."""
    device_id = wearables.device_id_for(device_key)
    doc = await fs_get(COL_WEARABLE_TOKENS, _token_doc_id(device_id, provider))
    if not doc:
        raise HTTPException(status_code=404, detail=f"{provider} is not connected")

    if not wearables.is_expired(doc):
        return doc

    if not doc.get("refresh_token"):
        raise HTTPException(status_code=401, detail=f"{provider} session expired — reconnect")
    try:
        fresh = await wearables.refresh_token(provider, doc["refresh_token"])
    except Exception as exc:
        logger.warning("refresh %s: %s", provider, exc)
        raise HTTPException(status_code=401, detail=f"{provider} session expired — reconnect")

    # Vendors that rotate refresh tokens invalidate the old one on use, so the
    # new one must be persisted before the next call or the link is dead.
    merged = {**doc, **{k: v for k, v in fresh.items() if v is not None}, "created_at": now_iso()}
    await fs_put(COL_WEARABLE_TOKENS, merged, doc_id=_token_doc_id(device_id, provider))
    return merged


@api.post("/wearable/sync/{provider}")
async def wearable_sync(provider: str, request: Request) -> Dict[str, Any]:
    """Pull recent daily rows from a connected vendor. Body: { device_key, days? }"""
    if provider not in wearables.ADAPTERS:
        raise HTTPException(status_code=404, detail="unknown provider")

    body = await request.json()
    device_key = (body.get("device_key") or "").strip()
    if len(device_key) < 16:
        raise HTTPException(status_code=400, detail="a device_key is required")

    days = max(1, min(int(body.get("days") or 30), 180))
    token = await _valid_token(device_key, provider)

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            rows = await wearables.fetch(
                provider, client, token["access_token"],
                start.isoformat(), end.isoformat(),
            )
        except PermissionError:
            raise HTTPException(status_code=401, detail=f"{provider} rejected the token — reconnect")
        except Exception as exc:
            logger.warning("sync %s: %s", provider, exc)
            raise HTTPException(status_code=502, detail=f"{provider} sync failed")

    device_id = wearables.device_id_for(device_key)
    for row in rows:
        doc_id = f"{device_id}_{provider}_{row['date']}"
        await fs_put(
            COL_WEARABLE_READINGS,
            {"id": doc_id, "created_at": now_iso(), "device_id": device_id, **row},
            doc_id=doc_id,
        )

    return {"ok": True, "provider": provider, "total": len(rows), "rows": rows}


async def fs_delete(col: str, doc_id: str) -> bool:
    """Best-effort single-document delete. Returns whether it succeeded."""
    try:
        await get_db().collection(col).document(doc_id).delete()
        return True
    except Exception as exc:
        logger.warning("delete %s/%s: %s", col, doc_id, exc)
        return False


@api.post("/wearable/disconnect/{provider}")
async def wearable_disconnect(provider: str, request: Request) -> Dict[str, Any]:
    """Forget a vendor's tokens. Body: { device_key, purge_readings?: bool }

    Disconnecting is a privacy action, so it never raises on partial failure:
    it reports exactly what it managed to remove. Throwing here would tell a
    user their data is still linked when the tokens are in fact already gone.
    """
    body = await request.json()
    device_key = (body.get("device_key") or "").strip()
    if len(device_key) < 16:
        raise HTTPException(status_code=400, detail="a device_key is required")

    device_id = wearables.device_id_for(device_key)
    revoked = await fs_delete(COL_WEARABLE_TOKENS, _token_doc_id(device_id, provider))

    purged, failed = 0, 0
    if body.get("purge_readings"):
        try:
            rows = await fs_list(COL_WEARABLE_READINGS, where={"device_id": device_id}, limit=365)
        except Exception as exc:
            logger.warning("disconnect list %s: %s", provider, exc)
            rows = []
        for r in rows:
            if r.get("source") != provider or not r.get("id"):
                continue
            if await fs_delete(COL_WEARABLE_READINGS, r["id"]):
                purged += 1
            else:
                failed += 1

    return {
        "ok": True,
        "provider": provider,
        "revoked": revoked,
        "purged_readings": purged,
        "failed_deletes": failed,
    }


@api.get("/wearable/status")
async def wearable_status(device_id: str = Query(...)) -> Dict[str, Any]:
    """Which vendors this device has linked. Never returns token material."""
    rows = await fs_list(COL_WEARABLE_TOKENS, where={"device_id": device_id}, limit=20)
    connected = [
        {
            "provider": r.get("provider"),
            "connected_at": r.get("created_at"),
            "expired": wearables.is_expired(r),
        }
        for r in rows
    ]
    return {"ok": True, "connected": connected}


# ── App wiring ────────────────────────────────────────────
app.include_router(api)

# ── Static frontend (React build baked into the container) ─
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT_DIR / "static")))
if STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as StarletteHTTPException

    class SPAStaticFiles(StaticFiles):
        """Serve the React build; fall back to index.html for client routes.

        index.html is served with no-cache so a redeploy is picked up immediately
        (the hashed JS/CSS assets stay long-cached — only the shell revalidates).
        """

        async def get_response(self, path: str, scope):
            try:
                resp = await super().get_response(path, scope)
            except StarletteHTTPException as ex:
                if ex.status_code == 404:
                    resp = await super().get_response("index.html", scope)
                else:
                    raise
            if path in ("", "/", "index.html") or path.endswith("index.html"):
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return resp

    app.mount("/", SPAStaticFiles(directory=str(STATIC_DIR), html=True), name="spa")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def _shutdown() -> None:
    try:
        if _db is not None:
            _db.close()
    except Exception:
        pass
