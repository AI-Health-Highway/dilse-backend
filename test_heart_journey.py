"""Integration checks for the verified heart-health workflow."""
import unittest
from collections import defaultdict
from copy import deepcopy
from unittest.mock import patch
import httpx
import server
from services.heart_journey import journey_summary, validate_inputs
from services.journey_routes import LEADS

INPUTS = {'age': 48, 'sex': 'male', 'region': 'india', 'ethnicity': 'indian', 'sbp': 138, 'bmi': 27.4, 'total_cholesterol': 5.5, 'hdl': 1.2, 'ldl': 3.7, 'commute_hours': 2.5, 'smoking': 'non', 'diabetes_type': 'none'}

class JourneyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = defaultdict(dict)
        self.user = {'id': 'user-1', 'phone': '+919876543210', 'phone_verified_at': server.now_iso(), 'assessment_consent': True, 'whatsapp_health_updates_consent': True}
        self.db[server.COL_PATIENTS]['user-1'] = deepcopy(self.user)
        self.db[server.COL_SNAPSHOTS]['baseline-scan'] = {'id': 'baseline-scan', 'patientId': 'user-1', 'created_at': server.now_iso(), 'fused': {'bpm': 72}}
        async def session(request, required=False):
            return deepcopy(self.db[server.COL_PATIENTS][self.user['id']])
        async def get(col, key):
            return deepcopy(self.db[col].get(key))
        async def put(col, value, doc_id=None):
            self.db[col][doc_id or value['id']] = deepcopy(value)
        async def rows(col, where=None, limit=None, **kwargs):
            result = [deepcopy(v) for v in self.db[col].values() if all(v.get(k) == value for k, value in (where or {}).items())]
            return result[:limit] if limit else result
        self.patches = [patch.object(server, '_session_patient', session), patch.object(server, 'fs_get', get), patch.object(server, 'fs_put', put), patch.object(server, 'fs_list', rows), patch.object(server, 'require_admin', lambda request: None)]
        for p in self.patches: p.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test')
    async def asyncTearDown(self):
        await self.client.aclose()
        for p in reversed(self.patches): p.stop()
    async def report(self):
        response = await self.client.post('/api/journey/assessment', json={'inputs': INPUTS, 'patientId': 'attacker', 'force': True})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()
    async def test_consent_precedes_profile_and_assessment(self):
        self.db[server.COL_PATIENTS]['user-1']['assessment_consent'] = False
        response = await self.client.put('/api/journey/profile', json=INPUTS)
        self.assertEqual(response.status_code, 403)
        response = await self.client.post('/api/journey/assessment', json={'inputs': INPUTS})
        self.assertEqual(response.status_code, 403)
        response = await self.client.put('/api/journey/consent', json={'consent': True, 'whatsappConsent': False})
        self.assertEqual(response.status_code, 200, response.text)
        response = await self.client.put('/api/journey/profile', json=INPUTS)
        self.assertEqual(response.status_code, 200)
    async def test_report_persistence_cycle_comparison_and_tenant(self):
        first = await self.report()
        self.assertEqual([p['day'] for p in first['journey']['plan']], [30, 60, 90])
        self.assertEqual(next(iter(self.db[server.COL_ASSESSMENTS].values()))['patientId'], 'user-1')
        second = await self.report()
        self.assertEqual(second['journey']['comparison']['previousHeartAge'], first['journey']['heartAge'])
        self.db[server.COL_HEALTH_REPORTS]['foreign'] = {'id': 'foreign', 'patientId': 'user-2', 'created_at': server.now_iso()}
        response = await self.client.get('/api/journey/reports/foreign')
        self.assertEqual(response.status_code, 404)
        response = await self.client.get('/api/journey/reports')
        self.assertTrue(all(r['patientId'] == 'user-1' for r in response.json()['rows']))
    async def test_report_requires_scan_and_reopening_reuses_result(self):
        self.db[server.COL_SNAPSHOTS].clear()
        response = await self.client.post('/api/journey/assessment', json={'inputs': INPUTS})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['detail']['code'], 'SCAN_REQUIRED')
        self.db[server.COL_SNAPSHOTS]['scan'] = {'id': 'scan', 'patientId': 'user-1', 'created_at': server.now_iso(), 'fused': {'bpm': 72}}
        first = await self.report()
        response = await self.client.post('/api/journey/assessment', json={'inputs': INPUTS})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['reportId'], first['reportId'])
        self.assertTrue(response.json()['cached'])
        self.db[server.COL_SNAPSHOTS]['foreign'] = {'id': 'foreign', 'patientId': 'user-2'}
        self.assertEqual((await self.client.get('/api/snapshots/foreign')).status_code, 404)

    async def test_lead_idempotency_permission_and_status_order(self):
        body = {'labId': 'clumax', 'location': 'Indiranagar', 'preferredAt': '2099-10-01T12:00:00+05:30', 'contactPermission': False, 'requestId': 'same-click'}
        response = await self.client.post('/api/journey/leads', json=body)
        self.assertEqual(response.status_code, 400)
        body['contactPermission'] = True
        first = await self.client.post('/api/journey/leads', json=body)
        second = await self.client.post('/api/journey/leads', json=body)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()['leadId'], second.json()['leadId'])
        self.assertEqual(len(self.db[LEADS]), 1)
        lead_id = first.json()['leadId']
        self.assertEqual((await self.client.patch('/api/admin/heart-leads/'+lead_id, json={'status': 'COMPLETED'})).status_code, 400)
        for status in ['CONTACTED', 'BOOKED', 'COMPLETED']:
            response = await self.client.patch('/api/admin/heart-leads/'+lead_id, json={'status': status})
            self.assertEqual(response.status_code, 200, response.text)
    async def test_daily_progress_and_validation(self):
        report = await self.report()
        route = '/api/journey/reports/'+report['reportId']+'/progress'
        self.assertEqual((await self.client.put(route, json={'completed': [0, 99]})).status_code, 400)
        self.assertEqual((await self.client.put(route, json={'completed': [0, 2]})).status_code, 200)
        self.assertEqual((await self.client.get(route)).json()['completed'], [0, 2])
    async def test_raw_scan_is_account_bound_and_requires_consent(self):
        raw = {'source': 'rppg', 'samples': [{'t': 1, 'r': 2, 'g': 3, 'b': 4}]}
        response = await self.client.post('/api/snapshot', json={'patientId': 'attacker', 'fused': {'bpm': 72}, 'rawMeasurements': raw})
        self.assertEqual(response.status_code, 200, response.text)
        scan = self.db[server.COL_SNAPSHOTS][response.json()['id']]
        self.assertEqual(scan['patientId'], 'user-1')
        self.assertEqual(scan['rawMeasurements'], raw)
        self.db[server.COL_PATIENTS]['user-1']['assessment_consent'] = False
        response = await self.client.post('/api/snapshot', json={'fused': {'bpm': 72}})
        self.assertEqual(response.status_code, 403)

class InputTests(unittest.TestCase):
    def test_reject_missing_or_impossible_age_and_pressure(self):
        for values in ({}, dict(INPUTS, age='nan'), dict(INPUTS, sbp=-1)):
            with self.assertRaises(ValueError): validate_inputs(values)
    def test_reference_age_and_risk_factors_use_inputs(self):
        inputs = validate_inputs(INPUTS)
        result = journey_summary(inputs, server.route_and_run(inputs))
        self.assertEqual(result['calendarAge'], 48)
        self.assertTrue(25 <= result['heartAge'] <= 84)
        self.assertIn('sbp', [f['key'] for f in result['factors']])
        self.assertIn('dailyTasks', result)
