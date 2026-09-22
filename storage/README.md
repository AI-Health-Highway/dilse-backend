# Storage Layer — Firestore migration notes

The Somatic backend currently persists data in **MongoDB** via Motor (`server.py`).
The user has indicated they want to migrate to **Firestore (documents) + Cloud
Storage (media)** on GCP. Credentials will be supplied later.

## Current state
- All reads/writes go through Motor against the configured `MONGO_URL`.
- Collections used today:
  - `assessments` (legacy, can be archived after migration)
  - `snapshots`   — scan readings (BPM/HRV/SpO2/quality/mode)
  - `journals`    — user reflections (if present)
  - `practice_logs` — practice taps (affirmation/journal/etc.)
  - `narratives`  — Mistral insight cache (also used by `/api/insight`)
  - `notifies`    — notify-me submissions
- API contract is documented in `server.py` and is stable: no frontend changes
  are required when switching backends.

## How to enable Firestore later
1. **Drop credentials** at `/app/backend/firebase-service-account.json`.
2. **Add env vars** to `/app/backend/.env`:
   ```
   DATA_BACKEND=firestore
   FIREBASE_SERVICE_ACCOUNT_PATH=/app/backend/firebase-service-account.json
   FIREBASE_STORAGE_BUCKET=<your-project>.appspot.com
   GCP_PROJECT_ID=<your-project-id>
   ```
3. **Install Firebase Admin SDK** (deferred to avoid bloating the image until
   credentials are in place):
   ```bash
   pip install firebase-admin google-cloud-storage
   pip freeze > /app/backend/requirements.txt
   ```
4. **Wire the adapter**. Build a `FirestoreRepo` with the same surface as the
   current Mongo collections (`save_snapshot`, `get_snapshot`, `list_snapshots`,
   `save_journal`, `list_journals`, `log_practice`, `get_streak`,
   `save_insight_cache`, `get_insight_cache`).
5. **Switch on the flag** — when `DATA_BACKEND=firestore`, replace the Motor
   collection handles with the Firestore adapter. The API responses stay the
   same.

## Suggested collections in Firestore
| Firestore collection | Doc shape |
|---|---|
| `snapshots`     | `{ id, created_at, mode, fused, face?, finger? }` |
| `journals`      | `{ id, created_at, text, prompt, snapshotId? }` |
| `practice_logs` | `{ id, created_at, kind, snapshotId?, ... }` |
| `narratives`    | `{ id, key, kind, created_at, text, vitals, state }` |
| `notifies`      | `{ id, created_at, feature, email }` |

## Cloud Storage paths (for future media)
- `users/{uid}/journals/{journalId}/{filename}` for journal audio/photo attachments
- Public access: never; signed URLs only.

> The current Mongo implementation will keep working until you finalize Firestore.
