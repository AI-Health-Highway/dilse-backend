"""
DilSay Phase 8 Backend API Test Suite
Tests new echo-centers endpoints + regression tests for all existing endpoints
"""

import requests
import sys
from datetime import datetime
from typing import Dict, Any, List

# Public endpoint from frontend/.env
BASE_URL = "https://rpulse-demo.preview.emergentagent.com/api"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

class DilSayAPITester:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failed_tests: List[Dict[str, Any]] = []

    def test(self, name: str, method: str, endpoint: str, expected_status: int = 200, 
             data: Dict = None, params: Dict = None, validate_fn=None) -> bool:
        """Run a single API test"""
        url = f"{BASE_URL}/{endpoint}"
        self.tests_run += 1
        
        print(f"\n{Colors.BLUE}[{self.tests_run}] Testing: {name}{Colors.END}")
        print(f"   {method} {endpoint}")
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, timeout=15)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=15)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=15)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check status code
            if response.status_code != expected_status:
                self._fail(name, f"Expected status {expected_status}, got {response.status_code}", response.text[:200])
                return False
            
            # Parse JSON
            try:
                json_data = response.json()
            except Exception as e:
                self._fail(name, f"Failed to parse JSON: {e}", response.text[:200])
                return False
            
            # Custom validation
            if validate_fn:
                validation_result = validate_fn(json_data)
                if validation_result is not True:
                    self._fail(name, f"Validation failed: {validation_result}", json_data)
                    return False
            
            self._pass(name)
            return True
            
        except requests.exceptions.Timeout:
            self._fail(name, "Request timeout (15s)", "")
            return False
        except Exception as e:
            self._fail(name, f"Exception: {str(e)}", "")
            return False

    def _pass(self, name: str):
        self.tests_passed += 1
        print(f"   {Colors.GREEN}✓ PASSED{Colors.END}")

    def _fail(self, name: str, reason: str, details: Any):
        self.tests_failed += 1
        self.failed_tests.append({"name": name, "reason": reason, "details": details})
        print(f"   {Colors.RED}✗ FAILED: {reason}{Colors.END}")
        if details:
            print(f"   {Colors.YELLOW}Details: {details}{Colors.END}")

    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"{Colors.BLUE}TEST SUMMARY{Colors.END}")
        print(f"{'='*70}")
        print(f"Total tests: {self.tests_run}")
        print(f"{Colors.GREEN}Passed: {self.tests_passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.tests_failed}{Colors.END}")
        
        if self.failed_tests:
            print(f"\n{Colors.RED}FAILED TESTS:{Colors.END}")
            for i, test in enumerate(self.failed_tests, 1):
                print(f"{i}. {test['name']}")
                print(f"   Reason: {test['reason']}")
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"\n{Colors.BLUE}Success Rate: {success_rate:.1f}%{Colors.END}")
        print(f"{'='*70}\n")


def main():
    tester = DilSayAPITester()
    
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}DilSay Phase 8 — Backend API Test Suite{Colors.END}")
    print(f"{Colors.BLUE}Testing: {BASE_URL}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")
    
    # ========================================================================
    # PHASE 8: NEW ECHO-CENTERS ENDPOINTS (Google Maps Integration)
    # ========================================================================
    
    print(f"\n{Colors.YELLOW}>>> PHASE 8: Echo Centers Endpoints (3 Partner Brands){Colors.END}")
    
    # Test 1: GET /api/echo-centers — returns exactly 3 partner brand cards
    tester.test(
        "GET /api/echo-centers — returns exactly 3 partner brand cards",
        "GET", "echo-centers",
        validate_fn=lambda d: (
            True if d.get("ok") and len(d.get("rows", [])) == 3 
            else f"Expected 3 partner cards, got {len(d.get('rows', []))}"
        )
    )
    
    # Test 2: GET /api/echo-centers — check brands metadata
    tester.test(
        "GET /api/echo-centers — includes brands metadata (aarthi, lalpath, clumax)",
        "GET", "echo-centers",
        validate_fn=lambda d: (
            True if d.get("brands") and "aarthi" in d.get("brands", {}) and "lalpath" in d.get("brands", {}) and "clumax" in d.get("brands", {})
            else "Missing aarthi, lalpath or clumax in brands metadata"
        )
    )
    
    # Test 3: GET /api/echo-centers — validate card structure
    def validate_card_structure(d):
        if not d.get("ok") or not d.get("rows"):
            return "Missing ok or rows"
        for card in d.get("rows", []):
            required = ["id", "brand", "displayName", "tagline", "supportPhone", "website", "mapsQuery", "logoInitial", "accentColor", "echoAvailable"]
            missing = [f for f in required if f not in card]
            if missing:
                return f"Card missing fields: {missing}"
        return True
    
    tester.test(
        "GET /api/echo-centers — each card has required fields",
        "GET", "echo-centers",
        validate_fn=validate_card_structure
    )
    
    # Test 4: GET /api/echo-centers?brand=aarthi — returns only Aarthi card
    tester.test(
        "GET /api/echo-centers?brand=aarthi — returns only Aarthi card (total=1)",
        "GET", "echo-centers",
        params={"brand": "aarthi"},
        validate_fn=lambda d: (
            True if d.get("ok") and d.get("total") == 1 and len(d.get("rows", [])) == 1 and d.get("rows", [])[0].get("brand") == "aarthi"
            else f"Expected 1 aarthi card, got total={d.get('total')}, rows={len(d.get('rows', []))}"
        )
    )
    
    # Test 5: GET /api/echo-centers?brand=lalpath — returns only Lalpath card
    tester.test(
        "GET /api/echo-centers?brand=lalpath — returns only Lalpath card (total=1)",
        "GET", "echo-centers",
        params={"brand": "lalpath"},
        validate_fn=lambda d: (
            True if d.get("ok") and d.get("total") == 1 and len(d.get("rows", [])) == 1 and d.get("rows", [])[0].get("brand") == "lalpath"
            else f"Expected 1 lalpath card, got total={d.get('total')}, rows={len(d.get('rows', []))}"
        )
    )
    
    # Test 6: GET /api/echo-centers?brand=clumax — returns only Clumax card
    tester.test(
        "GET /api/echo-centers?brand=clumax — returns only Clumax card (total=1)",
        "GET", "echo-centers",
        params={"brand": "clumax"},
        validate_fn=lambda d: (
            True if d.get("ok") and d.get("total") == 1 and len(d.get("rows", [])) == 1 and d.get("rows", [])[0].get("brand") == "clumax"
            else f"Expected 1 clumax card, got total={d.get('total')}, rows={len(d.get('rows', []))}"
        )
    )
    
    # Test 7: Validate Aarthi card details
    def validate_aarthi(d):
        if not d.get("rows") or len(d.get("rows")) != 1:
            return "Expected 1 row"
        card = d["rows"][0]
        if card.get("displayName") != "Aarthi Scans & Labs":
            return f"Wrong displayName: {card.get('displayName')}"
        if card.get("mapsQuery") != "Aarthi Scans and Labs":
            return f"Wrong mapsQuery: {card.get('mapsQuery')}"
        if card.get("supportPhone") != "+91-44-4297-4444":
            return f"Wrong supportPhone: {card.get('supportPhone')}"
        if not card.get("echoAvailable"):
            return "echoAvailable should be True"
        return True
    
    tester.test(
        "GET /api/echo-centers?brand=aarthi — validate Aarthi card details",
        "GET", "echo-centers",
        params={"brand": "aarthi"},
        validate_fn=validate_aarthi
    )
    
    # Test 8: Validate Clumax card details
    def validate_clumax(d):
        if not d.get("rows") or len(d.get("rows")) != 1:
            return "Expected 1 row"
        card = d["rows"][0]
        if card.get("displayName") != "Clumax Diagnostics":
            return f"Wrong displayName: {card.get('displayName')}"
        if card.get("mapsQuery") != "Clumax Diagnostics":
            return f"Wrong mapsQuery: {card.get('mapsQuery')}"
        if not card.get("echoAvailable"):
            return "echoAvailable should be True"
        return True
    
    tester.test(
        "GET /api/echo-centers?brand=clumax — validate Clumax card details",
        "GET", "echo-centers",
        params={"brand": "clumax"},
        validate_fn=validate_clumax
    )
    
    # ========================================================================
    # REGRESSION TESTS: Existing Endpoints
    # ========================================================================
    
    print(f"\n{Colors.YELLOW}>>> REGRESSION: Core Endpoints{Colors.END}")
    
    # Test 9: GET /api/ — root
    tester.test(
        "GET /api/ — root endpoint",
        "GET", "",
        validate_fn=lambda d: True if d.get("ok") and d.get("service") == "aisteth" else "Invalid root response"
    )
    
    # Test 10: GET /api/config
    tester.test(
        "GET /api/config — config endpoint",
        "GET", "config",
        validate_fn=lambda d: True if "mistralKey" in d else "Missing mistralKey in config"
    )
    
    # Test 11: POST /api/snapshot
    snapshot_data = {
        "fused": {"bpm": 72, "hrv_ms": 45, "spo2": 98, "brpm": 16, "quality": "strong"},
        "assessmentId": "test-assessment-123",
        "patientId": None,
        "face": {"result": {"bpm": 72}, "samples": []},
        "finger": None
    }
    tester.test(
        "POST /api/snapshot — save snapshot",
        "POST", "snapshot",
        data=snapshot_data,
        validate_fn=lambda d: True if d.get("ok") and d.get("id") else "Failed to save snapshot"
    )
    
    # Test 12: GET /api/snapshots
    tester.test(
        "GET /api/snapshots — list snapshots",
        "GET", "snapshots",
        validate_fn=lambda d: True if d.get("ok") and "rows" in d else "Invalid snapshots response"
    )
    
    # Test 13: GET /api/stats
    tester.test(
        "GET /api/stats — stats endpoint",
        "GET", "stats",
        validate_fn=lambda d: True if d.get("ok") and "totalScans" in d else "Invalid stats response"
    )
    
    # Test 14: GET /api/affirmations/today
    tester.test(
        "GET /api/affirmations/today — today's affirmation",
        "GET", "affirmations/today",
        validate_fn=lambda d: True if d.get("ok") and d.get("affirmation") else "Missing affirmation"
    )
    
    # Test 15: GET /api/streak
    tester.test(
        "GET /api/streak — streak endpoint",
        "GET", "streak",
        validate_fn=lambda d: True if d.get("ok") and "streak" in d else "Invalid streak response"
    )
    
    # Test 16: POST /api/practice
    tester.test(
        "POST /api/practice — log practice",
        "POST", "practice",
        data={"kind": "scan", "snapshotId": "test-snap-123"},
        validate_fn=lambda d: True if d.get("ok") else "Failed to log practice"
    )
    
    # Test 17: GET /api/journal
    tester.test(
        "GET /api/journal — list journal entries",
        "GET", "journal",
        validate_fn=lambda d: True if d.get("ok") and "rows" in d else "Invalid journal response"
    )
    
    # Test 18: POST /api/journal
    tester.test(
        "POST /api/journal — create journal entry",
        "POST", "journal",
        data={"text": "Test journal entry", "mood": "calm"},
        validate_fn=lambda d: True if d.get("ok") and d.get("row") else "Failed to create journal entry"
    )
    
    # Test 19: GET /api/health/models
    tester.test(
        "GET /api/health/models — list risk models",
        "GET", "health/models",
        validate_fn=lambda d: True if d.get("ok") and d.get("models") else "Invalid models response"
    )
    
    # Test 20: POST /api/health/risk
    risk_data = {
        "region": "india",
        "ethnicity": "indian",
        "age": 45,
        "sex": "male",
        "sbp": 130,
        "bmi": 26,
        "smoking": False,
        "diabetes_type": 0
    }
    tester.test(
        "POST /api/health/risk — compute risk",
        "POST", "health/risk",
        data=risk_data,
        validate_fn=lambda d: True if d.get("ok") and d.get("result") else "Failed to compute risk"
    )
    
    # Test 21: GET /api/patients
    tester.test(
        "GET /api/patients — list patients",
        "GET", "patients",
        validate_fn=lambda d: True if d.get("ok") and "rows" in d else "Invalid patients response"
    )
    
    # Test 22: POST /api/notify-me
    tester.test(
        "POST /api/notify-me — email capture",
        "POST", "notify-me",
        data={"feature": "auscultation", "email": f"test_{datetime.now().timestamp()}@example.com"},
        validate_fn=lambda d: True if d.get("ok") else "Failed to save email"
    )
    
    # Test 23: GET /api/privacy
    tester.test(
        "GET /api/privacy — privacy notice",
        "GET", "privacy",
        validate_fn=lambda d: True if d.get("ok") and d.get("principles") else "Invalid privacy response"
    )
    
    # Test 24: GET /api/assessments
    tester.test(
        "GET /api/assessments — list assessments",
        "GET", "assessments",
        validate_fn=lambda d: True if d.get("ok") and "rows" in d else "Invalid assessments response"
    )
    
    # Test 25: GET /api/health/reports
    tester.test(
        "GET /api/health/reports — list health reports",
        "GET", "health/reports",
        validate_fn=lambda d: True if d.get("ok") and "rows" in d else "Invalid health reports response"
    )
    
    # Print summary
    tester.print_summary()
    
    # Return exit code
    return 0 if tester.tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
