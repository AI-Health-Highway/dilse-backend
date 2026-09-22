import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import server


class _Request:
    async def json(self):
        return {
            "inputs": {
                "region": "india",
                "ethnicity": "indian",
                "age": 55,
                "sex": "male",
                "sbp": 145,
                "bmi": 26.5,
                "total_cholesterol": 5.5,
                "hdl": 1.2,
                "smoking": "non",
                "diabetes_type": "type2",
            },
            "vitals": {"bpm": 78, "hrv_ms": 45},
            "force": True,
        }


class _RateLimitedResponse:
    status_code = 429
    text = "rate limited"


class _RateLimitedClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, *_args, **_kwargs):
        return _RateLimitedResponse()


class HealthFallbackTests(unittest.TestCase):
    def test_rate_limit_returns_local_report(self):
        with (
            patch.object(server, "fs_get", new=AsyncMock(return_value=None)),
            patch.object(server, "fs_put", new=AsyncMock()),
            patch.object(server.httpx, "AsyncClient", return_value=_RateLimitedClient()),
        ):
            result = asyncio.run(server.health_analyze(_Request()))

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["analysis"]["tests"]), 3)
        self.assertEqual(len(result["analysis"]["lifestyle"]), 3)


if __name__ == "__main__":
    unittest.main()
