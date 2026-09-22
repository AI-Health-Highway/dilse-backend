"""Per-vendor fetch + normalize.

Every adapter has the same signature and returns the same shape:

    async def fetch(client, access_token, start_date, end_date) -> {date: row}

    row = {resting_hr, hrv_ms, spo2, steps, sleep_min, resp_rate}  (all optional)

`client` is injected rather than constructed, so tests drive these with a fake
transport and no network.

A deliberate design choice: extraction is *tolerant*. Vendors rename nested
fields between minor versions, and a KeyError that drops an entire sync is a
worse outcome than a missing metric for one day. Each extractor takes a list of
candidate key paths and returns the first plausible number it finds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import httpx

log = logging.getLogger("wearables.adapters")

# Physiological sanity bounds. Anything outside is treated as absent — vendors
# do emit zeros and sentinels, and a 0 bpm resting heart rate in a health app
# is worse than no reading at all.
BOUNDS = {
    "resting_hr": (25, 240),
    "hrv_ms": (1, 400),
    "spo2": (50, 100),
    "steps": (0, 200_000),
    "sleep_min": (1, 1440),
    "resp_rate": (4, 60),
}


def clamp(metric: str, value: Any) -> Optional[int]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = BOUNDS[metric]
    if not (lo <= v <= hi):
        return None
    return int(round(v))


def _dig(obj: Any, path: str) -> Any:
    """Dotted lookup that returns None instead of raising."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def first_of(obj: Any, paths: Iterable[str]) -> Any:
    for p in paths:
        v = _dig(obj, p)
        if v is not None:
            return v
    return None


def to_date(value: Any) -> Optional[str]:
    """Coerce the many date shapes vendors use into YYYY-MM-DD."""
    if value is None:
        return None
    if isinstance(value, dict):  # Google's {year, month, day}
        y, m, d = value.get("year"), value.get("month"), value.get("day")
        if y and m and d:
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        return None
    s = str(value)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def put(rows: Dict[str, Dict[str, Any]], date: Optional[str], metric: str, value: Any) -> None:
    """Write one metric into the daily bucket, if it survives clamping."""
    if not date:
        return
    v = clamp(metric, value)
    if v is None:
        return
    rows.setdefault(date, {})[metric] = v


def put_any(rows: Dict[str, Dict[str, Any]], date: Optional[str], metric: str,
            values: Iterable[Any]) -> None:
    """Write the first candidate that survives clamping.

    Distinct from `put(first_of(...))`: vendors emit in-band sentinels (Oura
    sends `lowest_heart_rate: 0` when the ring wasn't worn). Those are present
    but invalid, so "first non-null" picks the sentinel and silently loses the
    good fallback value. Validate each candidate instead of just the first.
    """
    if not date:
        return
    for value in values:
        v = clamp(metric, value)
        if v is not None:
            rows.setdefault(date, {})[metric] = v
            return


def _window(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    s = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    return s, e


async def _get(client: httpx.AsyncClient, url: str, token: str, **params) -> Dict[str, Any]:
    r = await client.get(
        url,
        params={k: v for k, v in params.items() if v is not None},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    if r.status_code == 401:
        raise PermissionError("token rejected")
    if r.status_code >= 400:
        raise RuntimeError(f"{url} -> {r.status_code}: {r.text[:200]}")
    return r.json()


# ── Oura ─────────────────────────────────────────────────────────────────

async def fetch_oura(client, token, start_date, end_date) -> Dict[str, Dict[str, Any]]:
    base = "https://api.ouraring.com/v2/usercollection"
    rows: Dict[str, Dict[str, Any]] = {}
    common = {"start_date": start_date, "end_date": end_date}

    # Sleep carries the heart metrics: lowest HR overnight is the best proxy
    # for resting HR, and Oura's average_hrv is already RMSSD in ms.
    try:
        sleep = await _get(client, f"{base}/sleep", token, **common)
        for it in sleep.get("data", []):
            d = to_date(it.get("day"))
            put_any(rows, d, "resting_hr", [it.get("lowest_heart_rate"), it.get("average_heart_rate")])
            put(rows, d, "hrv_ms", it.get("average_hrv"))
            put(rows, d, "resp_rate", it.get("average_breath"))
            secs = it.get("total_sleep_duration")
            if secs:
                put(rows, d, "sleep_min", float(secs) / 60.0)
    except Exception as e:  # one endpoint failing shouldn't kill the sync
        log.warning("oura sleep: %s", e)

    try:
        spo2 = await _get(client, f"{base}/daily_spo2", token, **common)
        for it in spo2.get("data", []):
            put_any(rows, to_date(it.get("day")), "spo2",
                    [_dig(it, "spo2_percentage.average"), it.get("spo2_percentage")])
    except Exception as e:
        log.warning("oura spo2: %s", e)

    try:
        act = await _get(client, f"{base}/daily_activity", token, **common)
        for it in act.get("data", []):
            put(rows, to_date(it.get("day")), "steps", it.get("steps"))
    except Exception as e:
        log.warning("oura activity: %s", e)

    return rows


# ── WHOOP ────────────────────────────────────────────────────────────────

def _whoop_hrv_to_ms(value: Any) -> Any:
    """WHOOP's `hrv_rmssd_milli` is reported in seconds in practice.

    A resting adult RMSSD is 20-120 ms, so a value under 5 is unambiguously
    seconds and must be scaled. This guards against the vendor changing units
    without breaking us either way.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v * 1000.0 if v < 5 else v


async def fetch_whoop(client, token, start_date, end_date) -> Dict[str, Dict[str, Any]]:
    base = "https://api.prod.whoop.com/developer/v2"
    start, end = _window(start_date, end_date)
    iso = {"start": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
           "end": end.strftime("%Y-%m-%dT%H:%M:%S.000Z")}
    rows: Dict[str, Dict[str, Any]] = {}

    try:
        rec = await _get(client, f"{base}/recovery", token, limit=25, **iso)
        for it in rec.get("records", []):
            d = to_date(first_of(it, ["created_at", "updated_at"]))
            score = it.get("score") or {}
            put(rows, d, "resting_hr", score.get("resting_heart_rate"))
            put(rows, d, "hrv_ms", _whoop_hrv_to_ms(score.get("hrv_rmssd_milli")))
            put(rows, d, "spo2", score.get("spo2_percentage"))
    except Exception as e:
        log.warning("whoop recovery: %s", e)

    try:
        sleep = await _get(client, f"{base}/activity/sleep", token, limit=25, **iso)
        for it in sleep.get("records", []):
            d = to_date(first_of(it, ["end", "start", "created_at"]))
            score = it.get("score") or {}
            put(rows, d, "resp_rate", score.get("respiratory_rate"))
            stage = score.get("stage_summary") or {}
            in_bed = stage.get("total_in_bed_time_milli")
            awake = stage.get("total_awake_time_milli") or 0
            if in_bed:
                put(rows, d, "sleep_min", (float(in_bed) - float(awake)) / 60000.0)
    except Exception as e:
        log.warning("whoop sleep: %s", e)

    return rows


# ── Garmin ───────────────────────────────────────────────────────────────

async def fetch_garmin(client, token, start_date, end_date) -> Dict[str, Dict[str, Any]]:
    base = "https://apis.garmin.com/wellness-api/rest"
    start, end = _window(start_date, end_date)
    # Garmin's summary endpoints page by upload time, in epoch seconds, and
    # cap each request at 24h of uploads.
    rows: Dict[str, Dict[str, Any]] = {}

    async def pull(path: str, handler) -> None:
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=1), end)
            try:
                data = await _get(
                    client, f"{base}/{path}", token,
                    uploadStartTimeInSeconds=int(cursor.timestamp()),
                    uploadEndTimeInSeconds=int(chunk_end.timestamp()),
                )
                for it in data if isinstance(data, list) else []:
                    handler(it)
            except Exception as e:
                log.warning("garmin %s: %s", path, e)
            cursor = chunk_end

    def on_daily(it):
        d = to_date(it.get("calendarDate"))
        put_any(rows, d, "resting_hr",
                [it.get("restingHeartRateInBeatsPerMinute"), it.get("restingHeartRateInBeats")])
        put(rows, d, "steps", it.get("steps"))
        put(rows, d, "resp_rate", it.get("avgWakingRespirationValue"))

    def on_sleep(it):
        d = to_date(it.get("calendarDate"))
        secs = it.get("durationInSeconds")
        if secs:
            put(rows, d, "sleep_min", float(secs) / 60.0)
        put(rows, d, "resp_rate", it.get("averageRespirationValue"))

    def on_hrv(it):
        put_any(rows, to_date(it.get("calendarDate")), "hrv_ms",
                [_dig(it, p) for p in ("lastNightAvg", "hrvSummary.lastNightAvg", "weeklyAvg")])

    def on_pulseox(it):
        put_any(rows, to_date(it.get("calendarDate")), "spo2",
                [_dig(it, p) for p in ("averageSpo2", "avgSpo2", "spo2Value")])

    await pull("dailies", on_daily)
    await pull("sleeps", on_sleep)
    await pull("hrv", on_hrv)
    await pull("pulseOx", on_pulseox)
    return rows


# ── Google Health API (v4) ───────────────────────────────────────────────
# Successor to the Fitbit Web API. Covers Fitbit trackers and Pixel Watch.

_GH_DATE_PATHS = [
    "interval.civilStartTime.date",
    "interval.startTime",
    "sampleTime.civilTime.date",
    "sampleTime.physicalTime",
    "civilStartTime.date",
    "date",
]

_GH_TYPES = [
    # (endpoint data type, response key, metric, candidate value paths)
    ("daily-resting-heart-rate", "dailyRestingHeartRate", "resting_hr",
     ["beatsPerMinute", "value", "restingHeartRate"]),
    ("daily-heart-rate-variability", "dailyHeartRateVariability", "hrv_ms",
     ["rmssd", "rmssdMilliseconds", "value"]),
    ("daily-oxygen-saturation", "dailyOxygenSaturation", "spo2",
     ["percentage", "avgPercentage", "average", "value"]),
    ("daily-respiratory-rate", "dailyRespiratoryRate", "resp_rate",
     ["breathsPerMinute", "value", "average"]),
]


async def fetch_google_health(client, token, start_date, end_date) -> Dict[str, Dict[str, Any]]:
    base = "https://health.googleapis.com/v4/users/me/dataTypes"
    rows: Dict[str, Dict[str, Any]] = {}

    for dtype, key, metric, value_paths in _GH_TYPES:
        # The filter parameter wants snake_case even though the path is kebab.
        field = key_to_snake(dtype)
        try:
            data = await _get(
                client, f"{base}/{dtype}/dataPoints", token,
                filter=f'{field}.interval.civil_start_time >= "{start_date}T00:00:00"',
                pageSize=200,
            )
        except Exception as e:
            log.warning("google_health %s: %s", dtype, e)
            continue

        for dp in data.get("dataPoints", []):
            inner = dp.get(key) or dp
            d = to_date(first_of(inner, _GH_DATE_PATHS)) or to_date(first_of(dp, _GH_DATE_PATHS))
            put_any(rows, d, metric, [_dig(inner, p) for p in value_paths])

    # Steps and sleep are shaped differently enough to handle separately.
    try:
        data = await _get(
            client, f"{base}/steps/dataPoints", token,
            filter=f'steps.interval.civil_start_time >= "{start_date}T00:00:00"',
            pageSize=1000,
        )
        totals: Dict[str, float] = {}
        for dp in data.get("dataPoints", []):
            inner = dp.get("steps") or {}
            d = to_date(first_of(inner, _GH_DATE_PATHS))
            count = first_of(inner, ["count", "countSum"])
            if d and count is not None:
                totals[d] = totals.get(d, 0) + float(count)
        for d, total in totals.items():
            put(rows, d, "steps", total)
    except Exception as e:
        log.warning("google_health steps: %s", e)

    try:
        data = await _get(
            client, f"{base}/sleep/dataPoints", token,
            filter=f'sleep.interval.civil_start_time >= "{start_date}T00:00:00"',
            pageSize=25,
        )
        for dp in data.get("dataPoints", []):
            inner = dp.get("sleep") or {}
            d = to_date(first_of(inner, ["interval.civilEndTime.date", "interval.endTime"] + _GH_DATE_PATHS))
            put_any(rows, d, "sleep_min",
                    [_dig(inner, "summary.minutesAsleep"), _dig(inner, "summary.minutesInSleepPeriod")])
    except Exception as e:
        log.warning("google_health sleep: %s", e)

    return rows


def key_to_snake(kebab: str) -> str:
    return kebab.replace("-", "_")


ADAPTERS = {
    "oura": fetch_oura,
    "whoop": fetch_whoop,
    "garmin": fetch_garmin,
    "google_health": fetch_google_health,
}


async def fetch(provider: str, client, token: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Run one provider's adapter and flatten to a sorted list of daily rows."""
    adapter = ADAPTERS.get(provider)
    if not adapter:
        raise KeyError(f"no adapter for provider: {provider}")
    rows = await adapter(client, token, start_date, end_date)
    out = [{"date": d, "source": provider, **metrics}
           for d, metrics in rows.items() if metrics]
    out.sort(key=lambda r: r["date"], reverse=True)
    return out
