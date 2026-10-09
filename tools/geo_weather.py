"""
Geo and weather tools (all free public APIs, no key needed).

Tools (order matters, the registry maps position -> tool number):
    1. weather_forecast    Open-Meteo current conditions + daily forecast (1-16 days, metric/imperial)
    2. geocode_place       Nominatim (OpenStreetMap) place name -> coordinates; Devanagari and other scripts work
    3. reverse_geocode     Nominatim coordinates -> address
    4. distance_calculator great-circle (haversine) distance computed locally; optional OSRM road distance
    5. country_info        REST Countries v3.1 facts (capital, population, currencies, languages, ...)

Notes:
  * Nominatim's usage policy requires an identifying User-Agent and max 1 request/second; both are honoured
    here. Set AGENT_CONTACT_EMAIL to put a contact address into the User-Agent (recommended).
  * Results are only what the upstream APIs returned; nothing is guessed. When an API is down or rate
    limited the tool returns an error instead of made-up data.
  * No optional packages needed.
"""

from __future__ import annotations

import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from ._common import ToolError, clamp, guard, no_results, ok_json, safe_request, truncate

_OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
_NOMINATIM = "https://nominatim.openstreetmap.org"
_OSRM = "https://router.project-osrm.org/route/v1/driving"
_RESTCOUNTRIES = "https://restcountries.com/v3.1"

_EARTH_RADIUS_KM = 6371.0088
_KM_PER_MILE = 1.609344
_KM_PER_NMI = 1.852
_MAX_QUERY = 200

# Nominatim policy: at most one request per second.
_MIN_INTERVAL = 1.1
_last_nominatim = 0.0

ATTRIBUTION_OSM = "Data (c) OpenStreetMap contributors, ODbL (via Nominatim)"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _num(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ToolError(f"{name} must be a number")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(x):
        raise ToolError(f"{name} must be a finite number")
    return x


def _coords(lat: Any, lon: Any, label: str = "") -> Tuple[float, float]:
    la, lo = _num(lat, f"{label}latitude"), _num(lon, f"{label}longitude")
    if not -90.0 <= la <= 90.0:
        raise ToolError(f"{label}latitude {la} is out of range (-90 to 90)")
    if not -180.0 <= lo <= 180.0:
        raise ToolError(f"{label}longitude {lo} is out of range (-180 to 180)")
    return la, lo


def _user_agent() -> str:
    ua = "agent_tools/1.0 (personal LangChain agent; Nominatim client)"
    contact = os.environ.get("AGENT_CONTACT_EMAIL", "").strip()
    if contact and re.fullmatch(r"[^\s@<>()]+@[^\s@<>()]+\.[^\s@<>()]+", contact):
        ua = ua[:-1] + f"; contact: {contact})"
    return ua


def _get_json(url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
              service: str, accept_404: bool = False, timeout: float = 20) -> Any:
    """GET + status/size/JSON validation. Returns None for an accepted 404."""
    hdrs = {"Accept": "application/json"}
    hdrs.update(headers or {})
    resp = safe_request("GET", url, params=params, headers=hdrs, timeout=timeout, max_bytes=2_000_000)
    if resp.status_code == 404 and accept_404:
        return None
    if resp.status_code == 429:
        raise ToolError(f"{service} rate limit reached (HTTP 429); wait a bit and retry")
    if resp.status_code != 200:
        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = str(body.get("reason") or body.get("message") or body.get("error") or "")
        except ValueError:
            detail = resp.text
        raise ToolError(f"{service} returned HTTP {resp.status_code}: {truncate(detail, 200)}")
    if getattr(resp, "truncated", False):
        raise ToolError(f"{service} response was too large to read")
    try:
        return resp.json()
    except ValueError:
        raise ToolError(f"{service} returned a non-JSON response")


def _nominatim_get(path: str, params: Dict[str, Any], language: str) -> Any:
    global _last_nominatim
    wait = _MIN_INTERVAL - (time.monotonic() - _last_nominatim)
    if wait > 0:
        time.sleep(min(wait, _MIN_INTERVAL))
    try:
        headers = {"User-Agent": _user_agent()}
        if language:
            headers["Accept-Language"] = language
        return _get_json(f"{_NOMINATIM}/{path}", params=params, headers=headers, service="Nominatim")
    finally:
        _last_nominatim = time.monotonic()


def _language(language: str) -> str:
    lang = (language or "").strip()
    if lang and not re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?", lang):
        raise ToolError("language must be a code like en, hi, hi-IN")
    return lang


def _place_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    addr = item.get("address") if isinstance(item.get("address"), dict) else {}
    out: Dict[str, Any] = {
        "name": item.get("name") or item.get("display_name"),
        "display_name": item.get("display_name"),
        "latitude": float(item["lat"]), "longitude": float(item["lon"]),
        "type": f"{item.get('category') or item.get('class') or ''}/{item.get('type') or ''}".strip("/"),
        "country": addr.get("country"), "country_code": (addr.get("country_code") or "").upper() or None,
        "state": addr.get("state"),
        "city": addr.get("city") or addr.get("town") or addr.get("village") or addr.get("suburb"),
        "postcode": addr.get("postcode"),
    }
    bb = item.get("boundingbox")
    if isinstance(bb, list) and len(bb) == 4:
        try:
            out["boundingbox_s_n_w_e"] = [float(v) for v in bb]
        except (TypeError, ValueError):
            pass
    if item.get("osm_type") and item.get("osm_id"):
        out["osm"] = f"{item['osm_type']}/{item['osm_id']}"
    return {k: v for k, v in out.items() if v not in (None, "")}


# ===========================================================================
# 1. weather_forecast
# ===========================================================================
_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle",
    57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain", 71: "Slight snow fall", 73: "Moderate snow fall",
    75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}
_CURRENT_VARS = ("temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,"
                 "wind_speed_10m,wind_direction_10m,is_day")
_DAILY_VARS = ("weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
               "precipitation_probability_max,wind_speed_10m_max,sunrise,sunset,uv_index_max")


def _compass(deg: Any) -> Optional[str]:
    if not isinstance(deg, (int, float)):
        return None
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return pts[int((deg % 360) / 22.5 + 0.5) % 16]


def _describe(code: Any) -> str:
    return _WMO.get(code, f"unknown WMO code {code}") if isinstance(code, int) else "unknown"


@tool
@guard
def weather_forecast(latitude: float = 999.0, longitude: float = 999.0, days: int = 3, units: str = "metric",
                     place: str = "") -> str:
    """Current weather plus a daily forecast from Open-Meteo (free, no key). Give coordinates, or a place name.

    Args:
        latitude: latitude in degrees, -90 to 90 (leave default when using place).
        longitude: longitude in degrees, -180 to 180 (leave default when using place).
        days: forecast days 1-16 (default 3; values outside are clamped).
        units: metric (C, km/h, mm) or imperial (F, mph, inch). Default metric.
        place: optional place name, e.g. "Darbhanga" or "दरभंगा", used only when coordinates are not given.
    """
    u = (units or "metric").strip().lower()
    if u not in {"metric", "imperial"}:
        raise ToolError("units must be 'metric' or 'imperial'")
    resolved = ""
    if latitude == 999.0 and longitude == 999.0:
        q = (place or "").strip()
        if not q:
            raise ToolError("provide latitude and longitude, or a place name")
        if len(q) > _MAX_QUERY:
            raise ToolError(f"place is longer than {_MAX_QUERY} characters")
        found = _nominatim_get("search", {"q": q, "format": "jsonv2", "limit": 1, "addressdetails": 1}, "")
        if not isinstance(found, list) or not found:
            return no_results(f"could not find a place called '{q}'; try geocode_place or give coordinates")
        s = _place_summary(found[0])
        latitude, longitude, resolved = s["latitude"], s["longitude"], str(s.get("display_name"))
    la, lo = _coords(latitude, longitude)
    n_days = clamp(days, 1, 16)

    params: Dict[str, Any] = {
        "latitude": la, "longitude": lo, "current": _CURRENT_VARS, "daily": _DAILY_VARS,
        "forecast_days": n_days, "timezone": "auto",
        "temperature_unit": "fahrenheit" if u == "imperial" else "celsius",
        "wind_speed_unit": "mph" if u == "imperial" else "kmh",
        "precipitation_unit": "inch" if u == "imperial" else "mm",
    }
    data = _get_json(_OPEN_METEO, params=params, service="Open-Meteo")
    if not isinstance(data, dict) or data.get("error"):
        raise ToolError(f"Open-Meteo error: {data.get('reason') if isinstance(data, dict) else 'bad response'}")
    cur, cu = data.get("current") or {}, data.get("current_units") or {}
    daily, du = data.get("daily") or {}, data.get("daily_units") or {}
    if not cur and not daily.get("time"):
        raise ToolError("Open-Meteo returned no weather data for these coordinates")

    def unit(units_map: Dict[str, Any], key: str) -> str:
        return str(units_map.get(key, ""))

    result: Dict[str, Any] = {
        "location": {"latitude": data.get("latitude", la), "longitude": data.get("longitude", lo),
                     "elevation_m": data.get("elevation"), "timezone": data.get("timezone"),
                     **({"resolved_from_place": resolved} if resolved else {})},
        "units": u,
    }
    if cur:
        result["current"] = {
            "time": cur.get("time"), "conditions": _describe(cur.get("weather_code")),
            "temperature": f"{cur.get('temperature_2m')}{unit(cu, 'temperature_2m')}",
            "feels_like": f"{cur.get('apparent_temperature')}{unit(cu, 'apparent_temperature')}",
            "humidity": f"{cur.get('relative_humidity_2m')}{unit(cu, 'relative_humidity_2m')}",
            "wind": f"{cur.get('wind_speed_10m')} {unit(cu, 'wind_speed_10m')} from "
                    f"{_compass(cur.get('wind_direction_10m')) or '?'}",
            "precipitation": f"{cur.get('precipitation')} {unit(cu, 'precipitation')}",
            "is_day": bool(cur.get("is_day")) if cur.get("is_day") is not None else None,
        }
    forecast: List[Dict[str, Any]] = []
    times = daily.get("time") or []

    def at(key: str, i: int) -> Any:
        vals = daily.get(key)
        return vals[i] if isinstance(vals, list) and i < len(vals) else None

    for i, day in enumerate(times):
        forecast.append({
            "date": day, "conditions": _describe(at("weather_code", i)),
            "max": f"{at('temperature_2m_max', i)}{unit(du, 'temperature_2m_max')}",
            "min": f"{at('temperature_2m_min', i)}{unit(du, 'temperature_2m_min')}",
            "precipitation_total": f"{at('precipitation_sum', i)} {unit(du, 'precipitation_sum')}",
            "precipitation_probability_max": f"{at('precipitation_probability_max', i)}"
                                             f"{unit(du, 'precipitation_probability_max')}",
            "wind_max": f"{at('wind_speed_10m_max', i)} {unit(du, 'wind_speed_10m_max')}",
            "uv_index_max": at("uv_index_max", i), "sunrise": at("sunrise", i), "sunset": at("sunset", i),
        })
    result["forecast"] = forecast
    result["source"] = "Open-Meteo (model forecast, not an official warning service)"
    return ok_json(result)


# ===========================================================================
# 2. geocode_place
# ===========================================================================
@tool
@guard
def geocode_place(query: str, limit: int = 5, language: str = "", country_codes: str = "") -> str:
    """Find coordinates for a place name or address using OpenStreetMap Nominatim. Works with Devanagari and other
    scripts. Rate limited to 1 request/second per the Nominatim usage policy.

    Args:
        query: place or address text, e.g. "C.M. Science College, Darbhanga" or "दरभंगा, बिहार".
        limit: maximum matches 1-10 (default 5).
        language: optional language for the result names, e.g. en or hi (default: local names).
        country_codes: optional comma-separated ISO 3166-1 alpha-2 codes to restrict results, e.g. "in" or "in,np".
    """
    q = " ".join((query or "").split())
    if not q:
        raise ToolError("query is empty")
    if len(q) > _MAX_QUERY:
        raise ToolError(f"query is longer than {_MAX_QUERY} characters")
    params: Dict[str, Any] = {"q": q, "format": "jsonv2", "limit": clamp(limit, 1, 10), "addressdetails": 1}
    cc = (country_codes or "").replace(" ", "").lower()
    if cc:
        if not re.fullmatch(r"[a-z]{2}(,[a-z]{2}){0,9}", cc):
            raise ToolError("country_codes must be 2-letter codes separated by commas, e.g. 'in,np'")
        params["countrycodes"] = cc
    data = _nominatim_get("search", params, _language(language))
    if not isinstance(data, list):
        raise ToolError("unexpected Nominatim response shape")
    if not data:
        return no_results(f"no place found for '{q}'. Try a less specific query (city and country).")
    results = [_place_summary(d) for d in data if isinstance(d, dict) and "lat" in d and "lon" in d]
    if not results:
        return no_results(f"no place with coordinates found for '{q}'")
    return ok_json({"query": q, "count": len(results), "results": results, "attribution": ATTRIBUTION_OSM})


# ===========================================================================
# 3. reverse_geocode
# ===========================================================================
_ZOOM = {"country": 3, "state": 5, "city": 10, "suburb": 14, "street": 16, "building": 18}


@tool
@guard
def reverse_geocode(latitude: float, longitude: float, detail: str = "street", language: str = "") -> str:
    """Turn coordinates into an address/place name using OpenStreetMap Nominatim (1 request/second policy).

    Args:
        latitude: latitude in degrees, -90 to 90.
        longitude: longitude in degrees, -180 to 180.
        detail: country, state, city, suburb, street or building (default street).
        language: optional language for names, e.g. en or hi.
    """
    la, lo = _coords(latitude, longitude)
    d = (detail or "street").strip().lower()
    if d not in _ZOOM:
        raise ToolError(f"detail must be one of: {', '.join(_ZOOM)}")
    params = {"lat": la, "lon": lo, "format": "jsonv2", "zoom": _ZOOM[d], "addressdetails": 1}
    data = _nominatim_get("reverse", params, _language(language))
    if not isinstance(data, dict):
        raise ToolError("unexpected Nominatim response shape")
    if data.get("error"):
        return no_results(f"no address found at {la}, {lo} ({data['error']}); it may be open sea or unmapped land")
    if "lat" not in data or "lon" not in data:
        raise ToolError("Nominatim response had no coordinates")
    out = _place_summary(data)
    if isinstance(data.get("address"), dict):
        out["address"] = {k: v for k, v in data["address"].items() if k != "ISO3166-2-lvl4" or v}
    out["requested"] = {"latitude": la, "longitude": lo}
    out["attribution"] = ATTRIBUTION_OSM
    return ok_json(out)


# ===========================================================================
# 4. distance_calculator
# ===========================================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    p1, p2, l1, dl = math.radians(lat1), math.radians(lat2), math.radians(lon1), math.radians(lon2 - lon1)
    bx, by = math.cos(p2) * math.cos(dl), math.cos(p2) * math.sin(dl)
    pm = math.atan2(math.sin(p1) + math.sin(p2), math.hypot(math.cos(p1) + bx, by))
    lm = l1 + math.atan2(by, math.cos(p1) + bx)
    return round(math.degrees(pm), 6), round((math.degrees(lm) + 540) % 360 - 180, 6)


def _osrm_road(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict[str, Any]:
    """Never raises: returns {'error': ...} so the straight-line result is still delivered."""
    label = ("OSRM public demo server, OpenStreetMap road data, CAR profile only; "
             "no live traffic, best-effort service")
    try:
        url = f"{_OSRM}/{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
        data = _get_json(url, params={"overview": "false"}, service="OSRM", timeout=20)
        if not isinstance(data, dict) or data.get("code") != "Ok" or not data.get("routes"):
            code = data.get("code") if isinstance(data, dict) else "bad response"
            return {"error": f"OSRM found no road route ({code}); the points may be on different landmasses",
                    "source": label}
        route = data["routes"][0]
        km = float(route["distance"]) / 1000
        return {"distance_km": round(km, 2), "distance_mi": round(km / _KM_PER_MILE, 2),
                "duration_minutes": round(float(route["duration"]) / 60, 1), "source": label}
    except (ToolError, KeyError, TypeError, ValueError) as e:
        return {"error": f"road distance unavailable: {e}", "source": label}
    except Exception as e:  # noqa: BLE001 - network failure must not hide the local result
        return {"error": f"road distance unavailable: {type(e).__name__}: {str(e)[:150]}", "source": label}


@tool
@guard
def distance_calculator(lat1: float, lon1: float, lat2: float, lon2: float, include_road: bool = False) -> str:
    """Distance between two coordinates. Always gives the great-circle (haversine, spherical Earth, accurate to
    roughly 0.5%) distance computed locally; optionally also the driving distance from OSRM (labelled, separate).

    Args:
        lat1: start latitude, -90 to 90.
        lon1: start longitude, -180 to 180.
        lat2: end latitude, -90 to 90.
        lon2: end longitude, -180 to 180.
        include_road: also query OSRM for car road distance and duration (needs internet; default False).
    """
    a_lat, a_lon = _coords(lat1, lon1, "start ")
    b_lat, b_lon = _coords(lat2, lon2, "end ")
    km = haversine_km(a_lat, a_lon, b_lat, b_lon)
    result: Dict[str, Any] = {
        "from": {"latitude": a_lat, "longitude": a_lon}, "to": {"latitude": b_lat, "longitude": b_lon},
        "straight_line": {
            "method": "haversine, mean Earth radius 6371.0088 km, computed locally (not a road distance)",
            "km": round(km, 3), "miles": round(km / _KM_PER_MILE, 3), "nautical_miles": round(km / _KM_PER_NMI, 3),
            "initial_bearing_deg": round(_bearing(a_lat, a_lon, b_lat, b_lon), 1) if km > 0 else None,
            "midpoint": dict(zip(("latitude", "longitude"), _midpoint(a_lat, a_lon, b_lat, b_lon))),
        },
    }
    if include_road:
        result["road"] = _osrm_road(a_lat, a_lon, b_lat, b_lon)
    return ok_json(result)


# ===========================================================================
# 5. country_info
# ===========================================================================
def _country_summary(c: Dict[str, Any]) -> Dict[str, Any]:
    name = c.get("name") or {}
    idd = c.get("idd") or {}
    suffixes = idd.get("suffixes") or []
    dial = (idd.get("root") or "") + (suffixes[0] if len(suffixes) == 1 else "")
    currencies = {k: f"{v.get('name')} ({v.get('symbol')})" if v.get("symbol") else v.get("name")
                  for k, v in (c.get("currencies") or {}).items() if isinstance(v, dict)}
    out = {
        "name": name.get("common"), "official_name": name.get("official"),
        "iso2": c.get("cca2"), "iso3": c.get("cca3"), "capital": c.get("capital"),
        "region": c.get("region"), "subregion": c.get("subregion"), "population": c.get("population"),
        "area_km2": c.get("area"), "languages": c.get("languages"), "currencies": currencies,
        "timezones": c.get("timezones"), "calling_code": dial or None, "internet_tld": c.get("tld"),
        "borders": c.get("borders"), "landlocked": c.get("landlocked"), "independent": c.get("independent"),
        "un_member": c.get("unMember"), "driving_side": (c.get("car") or {}).get("side"),
        "start_of_week": c.get("startOfWeek"), "lat_lng": c.get("latlng"),
        "flag_emoji": c.get("flag"), "flag_png": (c.get("flags") or {}).get("png"),
        "google_maps": (c.get("maps") or {}).get("googleMaps"),
    }
    return {k: v for k, v in out.items() if v not in (None, [], {}, "")}


@tool
@guard
def country_info(country: str, full_text: bool = False) -> str:
    """Facts about a country from REST Countries v3.1: capital, population, area, languages, currencies,
    timezones, calling code, borders. Accepts a name (English or native, e.g. India, Nepal, भारत) or an ISO code.

    Args:
        country: country name, or ISO 3166 alpha-2/alpha-3 code such as IN or IND.
        full_text: require an exact full-name match instead of partial matching (default False).
    """
    q = " ".join((country or "").split())
    if not q:
        raise ToolError("country is empty")
    if len(q) > 100:
        raise ToolError("country is longer than 100 characters")
    if re.search(r"[/\\?#%]", q):
        raise ToolError("country contains invalid characters")
    enc = urllib.parse.quote(q, safe="")
    data: Any = None
    if re.fullmatch(r"[A-Za-z]{2,3}", q):
        data = _get_json(f"{_RESTCOUNTRIES}/alpha/{enc.upper()}", service="REST Countries", accept_404=True)
        if isinstance(data, dict):
            data = [data]
    if not data:
        data = _get_json(f"{_RESTCOUNTRIES}/name/{enc}", params={"fullText": "true"} if full_text else None,
                         service="REST Countries", accept_404=True)
    if not isinstance(data, list) or not data:
        return no_results(f"no country matches '{q}'. Try the English common name or an ISO code.")
    countries = [c for c in data if isinstance(c, dict)]
    ql = q.casefold()

    def exact(c: Dict[str, Any]) -> bool:
        n = c.get("name") or {}
        return ql in {str(x).casefold() for x in (n.get("common"), n.get("official"), c.get("cca2"), c.get("cca3"))}

    best = next((c for c in countries if exact(c)), countries[0])
    result: Dict[str, Any] = {"country": _country_summary(best), "source": "REST Countries v3.1"}
    others = [(c.get("name") or {}).get("common") for c in countries if c is not best][:8]
    if others:
        result["other_matches"] = others
        if best is countries[0] and not exact(best):
            result["note"] = "partial name match; the first result was used, check other_matches if it is wrong"
    return ok_json(result)


GEO_WEATHER_TOOLS = [weather_forecast, geocode_place, reverse_geocode, distance_calculator, country_info]
