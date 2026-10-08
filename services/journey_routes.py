"""Authenticated assessment, plan and lead workflow on existing storage."""
from datetime import datetime, timedelta, timezone
import hashlib
import uuid

from fastapi import HTTPException, Request

from services.heart_journey import journey_summary, validate_inputs

CONSENTS = "assessment_consents"
LEADS = "heart_check_leads"
PROGRESS = "heart_plan_progress"


def register_journey_routes(api, backend):
    s = backend

    async def patient(request, consent=True):
        row = await s._session_patient(request, required=True)
        if consent and not row.get("assessment_consent"):
            raise HTTPException(403, "Consent is required before collecting assessment data")
        return row

    @api.get("/journey/profile")
    async def get_profile(request: Request):
        row = await patient(request, consent=False)
        return {"ok": True, "consent": bool(row.get("assessment_consent")), "profile": row.get("health_profile"), "whatsappConsent": bool(row.get("whatsapp_health_updates_consent"))}

    @api.put("/journey/consent")
    async def consent(request: Request):
        row = await patient(request, consent=False)
        body = await request.json()
        if not isinstance(body.get("consent"), bool):
            raise HTTPException(400, "Consent choices must be true or false")
        record = {"id": str(uuid.uuid4()), "patient_id": row["id"], "consent": body["consent"], "version": s.CONSENT_VERSION, "created_at": s.now_iso(), "source": "onboarding"}
        await s.fs_put(CONSENTS, record)
        row.update(assessment_consent=body["consent"], assessment_consent_at=s.now_iso(), assessment_consent_version=s.CONSENT_VERSION)
        await s.fs_put(s.COL_PATIENTS, row)
        return {"ok": True, "consent": body["consent"]}

    @api.put("/journey/profile")
    async def save_profile(request: Request):
        row = await patient(request)
        try:
            inputs = validate_inputs(await request.json())
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None
        row.update(health_profile=inputs, age=inputs["age"], sex=inputs["sex"], updated_at=s.now_iso())
        await s.fs_put(s.COL_PATIENTS, row)
        return {"ok": True, "profile": inputs}

    @api.post("/journey/assessment")
    async def assessment(request: Request):
        row = await patient(request)
        body = await request.json()
        try:
            inputs = validate_inputs(body.get("inputs") or row.get("health_profile") or {})
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None
        risk = s.route_and_run(inputs)
        journey = journey_summary(inputs, risk)
        previous = await s.fs_list(s.COL_HEALTH_REPORTS, where={"patientId": row["id"]}, order_by=None)
        previous.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        last = next((r for r in previous if r.get("journey")), None)
        if last:
            journey["comparison"] = {"previousDate": last["created_at"], "previousHeartAge": last["journey"]["heartAge"], "currentHeartAge": journey["heartAge"], "previousRisk": last["primaryScore"], "currentRisk": risk["primary"]["score"], "previousBand": last["primaryBand"], "currentBand": risk["primary"]["band"], "sameModel": last.get("primaryModel") == risk["primary"]["modelUsed"]}
        # Select only this verified user's latest scan, never client-supplied vitals.
        snapshots = await s.fs_list(s.COL_SNAPSHOTS, where={"patientId": row["id"]}, order_by=None)
        snapshots.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        scan = snapshots[0] if snapshots else {}
        if not scan:
            raise HTTPException(409, {"code": "SCAN_REQUIRED", "message": "Complete a face or finger scan before viewing your first report"})
        if last and not body.get("force") and last.get("inputs") == inputs and last.get("snapshotId") == scan.get("id"):
            return {"ok": True, "cached": True, "reportId": last["id"], "risk": last["risk"], "journey": last["journey"], "analysis": last["analysis"], "snapshotId": scan.get("id")}
        primary = risk["primary"]
        journey["riskHistory"] = [
            {"date": item["created_at"], "score": item["primaryScore"]}
            for item in reversed(previous[:19])
            if item.get("primaryModel") == primary["modelUsed"]
        ] + [{"date": s.now_iso(), "score": primary["score"]}]
        report = {"id": str(uuid.uuid4()), "patientId": row["id"], "created_at": s.now_iso(), "inputs": inputs, "vitals": scan.get("fused", {}), "snapshotId": scan.get("id"), "risk": risk, "journey": journey, "primaryModel": primary["modelUsed"], "primaryScore": primary["score"], "primaryBand": primary["band"], "analysis": s._fallback_health_analysis(risk)}
        await s.fs_put(s.COL_HEALTH_REPORTS, report)
        await s.fs_put(s.COL_ASSESSMENTS, {**report, "assessment_date": report["created_at"], "assessment_consent_version": row.get("assessment_consent_version")})
        row.update(health_profile=inputs, age=inputs["age"], sex=inputs["sex"], assessment_date=report["created_at"], updated_at=s.now_iso())
        await s.fs_put(s.COL_PATIENTS, row)
        return {"ok": True, "reportId": report["id"], "risk": risk, "journey": journey, "analysis": report["analysis"], "snapshotId": scan.get("id")}

    @api.get("/journey/reports")
    async def reports(request: Request):
        row = await patient(request)
        rows = await s.fs_list(s.COL_HEALTH_REPORTS, where={"patientId": row["id"]}, order_by=None)
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return {"ok": True, "rows": rows[:100]}

    async def owned_report(request, report_id):
        row = await patient(request)
        report = await s.fs_get(s.COL_HEALTH_REPORTS, report_id)
        if not report or report.get("patientId") != row["id"]:
            raise HTTPException(404, "Report not found")
        return row, report

    @api.get("/journey/reports/{report_id}")
    async def report(request: Request, report_id: str):
        _, value = await owned_report(request, report_id)
        return {"ok": True, "row": value}

    @api.get("/journey/reports/{report_id}/progress")
    async def progress(request: Request, report_id: str):
        row, _ = await owned_report(request, report_id)
        day = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
        record = await s.fs_get(PROGRESS, f"{row['id']}:{report_id}:{day}")
        return {"ok": True, "completed": (record or {}).get("completed", [])}

    @api.put("/journey/reports/{report_id}/progress")
    async def update_progress(request: Request, report_id: str):
        row, report = await owned_report(request, report_id)
        body = await request.json()
        completed = body.get("completed")
        count = len((report.get("journey") or {}).get("dailyTasks", []))
        if not isinstance(completed, list) or any(type(i) is not int or not 0 <= i < count for i in completed):
            raise HTTPException(400, "Invalid task selection")
        day = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
        record = {"id": f"{row['id']}:{report_id}:{day}", "patient_id": row["id"], "report_id": report_id, "date": day, "completed": sorted(set(completed)), "updated_at": s.now_iso()}
        await s.fs_put(PROGRESS, record)
        return {"ok": True, "completed": record["completed"]}

    @api.post("/journey/leads")
    async def create_lead(request: Request):
        row = await patient(request)
        body = await request.json()
        labs = {p["id"]: p for p in s.list_partners()}
        if body.get("labId") not in labs or body.get("contactPermission") is not True:
            raise HTTPException(400, "Select a lab and allow contact for this request")
        location = str(body.get("location") or "").strip()
        if not location or len(location) > 200:
            raise HTTPException(400, "Enter a preferred Bengaluru location")
        try:
            slot = datetime.fromisoformat(str(body.get("preferredAt") or ""))
            if slot.tzinfo is None or slot <= datetime.now(timezone.utc):
                raise ValueError()
        except ValueError:
            raise HTTPException(400, "Select a future date and time") from None
        report_id = body.get("reportId")
        if report_id:
            await owned_report(request, report_id)
        # Stable idempotency key prevents double-clicks creating duplicate leads.
        key = str(body.get("requestId") or "")
        if not key or len(key) > 100:
            raise HTTPException(400, "A requestId is required")
        lead_id = hashlib.sha256(f"{row['id']}:{key}".encode()).hexdigest()
        existing = await s.fs_get(LEADS, lead_id)
        if existing:
            return {"ok": True, "leadId": lead_id, "status": existing["status"]}
        record = {"id": lead_id, "patient_id": row["id"], "phone": row["phone"], "lab_id": body["labId"], "lab_name": labs[body["labId"]]["displayName"], "location": location, "city": "Bengaluru", "preferred_at": slot.isoformat(), "contact_permission": True, "contact_permission_at": s.now_iso(), "report_id": report_id, "status": "NEW", "created_at": s.now_iso(), "updated_at": s.now_iso()}
        await s.fs_put(LEADS, record)
        return {"ok": True, "leadId": lead_id, "status": "NEW"}

    @api.get("/journey/leads")
    async def leads(request: Request):
        row = await patient(request)
        rows = await s.fs_list(LEADS, where={"patient_id": row["id"]}, order_by=None)
        return {"ok": True, "rows": rows}

    @api.get("/admin/heart-leads")
    async def team_leads(request: Request):
        s.require_admin(request)
        return {"ok": True, "rows": await s.fs_list(LEADS, limit=200)}

    @api.patch("/admin/heart-leads/{lead_id}")
    async def lead_status(request: Request, lead_id: str):
        s.require_admin(request)
        record = await s.fs_get(LEADS, lead_id)
        if not record:
            raise HTTPException(404, "Lead not found")
        body = await request.json()
        next_status = {"NEW": "CONTACTED", "CONTACTED": "BOOKED", "BOOKED": "COMPLETED"}.get(record["status"])
        if body.get("status") != next_status:
            raise HTTPException(400, "Use NEW → CONTACTED → BOOKED → COMPLETED")
        record.update(status=next_status, updated_at=s.now_iso())
        await s.fs_put(LEADS, record)
        return {"ok": True, "row": record}

