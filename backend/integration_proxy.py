import os
import httpx
from dataclasses import dataclass
from typing import Any, Dict, Optional

USER_AGENT = "SeismoGuard/1.0 (+https://local)"
TIMEOUT = 15.0

PROVIDERS = {
    # USGS realtime feeds
    "usgs_realtime": {
        "base": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/",
        "type": "json",
        "endpoints": {
            # e.g., '4.5_day.geojson'
        },
    },
    # IRIS FDSN Event service
    "iris_event": {
        "base": "https://service.iris.edu/fdsnws/event/1/query",
        "type": "json",
    },
    # EMSC (Seismic Portal) FDSN Event
    "emsc_event": {
        "base": "https://www.seismicportal.eu/fdsnws/event/1/query",
        "type": "json",
    },
}

@dataclass
class FetchResult:
    ok: bool
    provider: str
    endpoint: Optional[str]
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None


async def fetch(provider: str, endpoint: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> FetchResult:
    provider = (provider or "").strip()
    if provider not in PROVIDERS:
        return FetchResult(False, provider, endpoint, None, f"Unknown provider: {provider}")

    conf = PROVIDERS[provider]
    base = conf["base"]
    kind = conf["type"]

    # Build URL
    if endpoint:
        url = base + endpoint if base.endswith('/') else base + '/' + endpoint
    else:
        url = base

    # Sensible defaults for FDSN event queries (iris/emsc)
    if provider in ("iris_event", "emsc_event"):
        if not params:
            params = {}
        params.setdefault("format", "json")
        params.setdefault("minmagnitude", 4.5)
        # Last 1 day
        from datetime import datetime, timedelta
        end = datetime.utcnow()
        start = end - timedelta(days=1)
        params.setdefault("starttime", start.strftime('%Y-%m-%dT%H:%M:%S'))
        params.setdefault("endtime", end.strftime('%Y-%m-%dT%H:%M:%S'))

    headers = {"User-Agent": USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            ct = resp.headers.get("content-type", "")
            if kind == "json" and "json" in ct:
                return FetchResult(resp.is_success, provider, endpoint, resp.json(), None if resp.is_success else resp.text, resp.status_code)
            else:
                # Return text if not JSON
                return FetchResult(resp.is_success, provider, endpoint, resp.text, None if resp.is_success else resp.text, resp.status_code)
    except Exception as exc:
        return FetchResult(False, provider, endpoint, None, str(exc), None)
