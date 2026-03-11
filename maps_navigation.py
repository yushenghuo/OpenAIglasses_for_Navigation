# maps_navigation.py
# 步行导航：高德 / Google（Geocoding + Directions），统一路线结构供播报与 map matching
# 从 darksight/navigation/amapNavigation.ts 迁移

import os
import re
import math
import logging
from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)

AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

# --------- 坐标系 WGS-84 -> GCJ-02（高德用）---------
A = 6378245.0
EE = 0.00669342162296594323


def _out_of_china(lng: float, lat: float) -> bool:
    return lng < 72.004 or lng > 137.8347 or lat < 0.8293 or lat > 55.8271


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin((y / 3.0) * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin((y / 12.0) * math.pi) + 320 * math.sin((y * math.pi) / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin((x / 3.0) * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin((x / 12.0) * math.pi) + 300.0 * math.sin((x / 30.0) * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lng: float, lat: float) -> Tuple[float, float]:
    if _out_of_china(lng, lat):
        return (lng, lat)
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    rad_lat = (lat / 180.0) * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrt_magic) * math.pi)
    dlng = (dlng * 180.0) / ((A / sqrt_magic) * math.cos(rad_lat) * math.pi)
    return (lng + dlng, lat + dlat)


# --------- Provider ---------
def _detect_default_provider() -> Literal["amap", "google"]:
    """
    决定默认导航 Provider 的策略：
    1) 若显式设置 NAV_PROVIDER，则以此为准（amap / google）；
    2) 否则若环境里存在 AMAP_WEB_SERVICE_KEY / AMAP_API_KEY，则优先用高德；
    3) 否则回退到 google。
    """
    nav_env = os.getenv("NAV_PROVIDER")
    if nav_env:
        return "google" if nav_env.lower() == "google" else "amap"
    amap_key = (
        os.getenv("AMAP_WEB_SERVICE_KEY")
        or os.getenv("AMAP_API_KEY")
        or "055ab3e6af570092bc63ae7e2597bc53"
    ).strip()
    if amap_key:
        return "amap"
    return "google"


_navigation_provider: Literal["amap", "google"] = _detect_default_provider()


def set_navigation_provider(provider: Literal["amap", "google"]) -> None:
    global _navigation_provider
    _navigation_provider = provider


def get_navigation_provider() -> Literal["amap", "google"]:
    return _navigation_provider


def get_amap_api_key() -> str:
    return (os.getenv("AMAP_WEB_SERVICE_KEY") or os.getenv("AMAP_API_KEY") or "").strip()


def get_navigation_api_key() -> str:
    if _navigation_provider == "google":
        return (
            os.getenv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY")
            or os.getenv("GOOGLE_MAPS_API_KEY")
            or "AIzaSyDHuOg-6uZxPyEZrfHh5DhLxUWomTjyfdk"
        ).strip()
    return get_amap_api_key()


# --------- 数据结构 ---------
@dataclass
class AmapStep:
    instruction: str
    road: str
    distance: str
    duration: str
    polyline: str


@dataclass
class AmapWalkingRoute:
    origin: Tuple[float, float]  # (lng, lat)
    destination: Tuple[float, float]
    distance: float
    duration: float
    steps: List[AmapStep]


@dataclass
class RoutePolylinePrecomputed:
    points: List[Tuple[float, float]]  # [(lng, lat), ...]
    cum_distances: List[float]


@dataclass
class MapMatchResult:
    matched_point: Tuple[float, float]
    matched_s: float
    matched_index: int
    off_route_distance: float


@dataclass
class AmapInstructionResult:
    text: str
    step_index: int
    kind: Optional[str]  # 'straight'|'turn_approach'|'turn_now'|'arrived'|'cross'|'cross_approach'


@dataclass
class AmapDisplayInfo:
    instruction_text: str
    distance_to_next_m: float
    turn_type: str  # 'straight'|'left'|'right'|'arrived'|'cross'
    road_name: str


# 工业级常量（与 darksight 一致）
OFF_ROUTE_DISTANCE_M = 25
MAX_ACCURACY_FOR_OFF_ROUTE_M = 25
ACCURACY_TOO_BAD_M = 50
OFF_ROUTE_CONSECUTIVE_FOR_REPLAN = 6
MIN_PROGRESS_BEFORE_OFF_ROUTE_M = 30
REPLAN_COOLDOWN_MS = 60_000
MAP_MATCH_WINDOW = 80
MAP_MATCH_MAX_BACKWARD_M = 12
DIST_STRAIGHT_M = 200
DIST_APPROACH_M = 80
DIST_TURN_NOW_M = 5
MIN_SAME_INSTRUCTION_INTERVAL_MS = 20000
MIN_DISTANCE_DELTA_M = 30


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lng1, lat1 = a
    lng2, lat2 = b
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    x = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))
    return R * c


def _parse_polyline(polyline: str) -> List[Tuple[float, float]]:
    out = []
    for s in polyline.split(";"):
        s = s.strip()
        if not s:
            continue
        parts = s.split(",")
        if len(parts) >= 2:
            out.append((float(parts[0]), float(parts[1])))
    return out


def _decode_google_polyline(encoded: str) -> List[Tuple[float, float]]:
    points = []
    index = 0
    lat, lng = 0, 0
    n = len(encoded)
    while index < n:
        shift, result = 0, 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else result >> 1
        lat += dlat
        shift, result = 0, 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else result >> 1
        lng += dlng
        points.append((lng / 1e5, lat / 1e5))
    return points


def precompute_route_polyline(route: AmapWalkingRoute) -> Optional[RoutePolylinePrecomputed]:
    points: List[Tuple[float, float]] = []
    for step in route.steps:
        pts = _parse_polyline(step.polyline)
        for i, p in enumerate(pts):
            if i == 0 and points and points[-1] == p:
                continue
            points.append(p)
    if len(points) < 2:
        return None
    cum = [0.0]
    for i in range(1, len(points)):
        cum.append(cum[-1] + _haversine_m(points[i - 1], points[i]))
    return RoutePolylinePrecomputed(points=points, cum_distances=cum)


def _project_point_onto_segment(
    g: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[Tuple[float, float], float, float]:
    glng, glat = g
    alng, alat = a
    blng, blat = b
    dlat = blat - alat
    dlng = blng - alng
    denom = dlat * dlat + dlng * dlng
    t = ((glat - alat) * dlat + (glng - alng) * dlng) / denom if denom != 0 else 0
    t = max(0, min(1, t))
    q = (alng + t * dlng, alat + t * dlat)
    dist = _haversine_m(g, q)
    return (q, t, dist)


def match_position_to_route(
    precomputed: RoutePolylinePrecomputed,
    last_index: int,
    last_s: float,
    gps_point: Tuple[float, float],
) -> MapMatchResult:
    points = precomputed.points
    cum = precomputed.cum_distances
    n = len(points)
    low = max(0, last_index - MAP_MATCH_WINDOW)
    high = min(n - 1, last_index + MAP_MATCH_WINDOW)
    best_dist = 1e9
    best_q = gps_point
    best_s = last_s
    best_index = last_index
    for i in range(low, high):
        a, b = points[i], points[i + 1]
        q, t, dist_m = _project_point_onto_segment(gps_point, a, b)
        seg_len = cum[i + 1] - cum[i]
        s_q = cum[i] + t * seg_len
        if s_q < last_s - MAP_MATCH_MAX_BACKWARD_M:
            continue
        if dist_m < best_dist:
            best_dist = dist_m
            best_q = q
            best_s = s_q
            best_index = i
    return MapMatchResult(
        matched_point=best_q,
        matched_s=best_s,
        matched_index=best_index,
        off_route_distance=best_dist,
    )


def _get_turn_type(instruction: str) -> Optional[str]:
    t = instruction.lower()
    if re.search(r"过马路|人行横道|过街|cross|crossing|cross the", t):
        return "cross"
    if re.search(r"左|向左|turn left|left onto|slight left", t):
        return "left"
    if re.search(r"右|向右|turn right|right onto|slight right", t):
        return "right"
    if re.search(r"直行|沿|straight|head |continue", t):
        return "straight"
    return None


def get_instruction_for_position(
    route: AmapWalkingRoute,
    position: Tuple[float, float],
    last_spoken: str,
    last_spoken_time: float,
    last_step_index: int,
) -> Optional[AmapInstructionResult]:
    import time
    now_ms = time.time() * 1000
    steps = route.steps
    if not steps:
        return None
    plng, plat = position
    current_step_index = 0
    min_dist = 1e9
    dist_to_next_turn_m = 0.0
    for i, step in enumerate(steps):
        pts = _parse_polyline(step.polyline)
        if not pts:
            continue
        step_start, step_end = pts[0], pts[-1]
        d_end = _haversine_m((plng, plat), step_end)
        if d_end < min_dist:
            min_dist = d_end
            current_step_index = i
            dist_to_next_turn_m = d_end
    step = steps[current_step_index]
    step_dist_m = float(step.distance) if step.distance else 0
    turn_type = _get_turn_type(step.instruction)
    is_last_step = current_step_index >= len(steps) - 1
    if is_last_step and step_dist_m < 20 and min_dist < 25:
        return AmapInstructionResult(
            text="You have arrived at your destination.",
            step_index=current_step_index,
            kind="arrived",
        )
    if turn_type == "cross":
        if dist_to_next_turn_m <= DIST_TURN_NOW_M:
            return AmapInstructionResult(
                text="Cross the road now.",
                step_index=current_step_index,
                kind="cross",
            )
        if dist_to_next_turn_m <= DIST_APPROACH_M:
            text = f"In about {int(round(dist_to_next_turn_m))} meters, cross the road."
            if text != last_spoken or (now_ms - last_spoken_time) >= MIN_SAME_INSTRUCTION_INTERVAL_MS:
                return AmapInstructionResult(text=text, step_index=current_step_index, kind="cross_approach")
        return None
    if turn_type in ("left", "right"):
        if dist_to_next_turn_m <= DIST_TURN_NOW_M:
            text = "Turn left." if turn_type == "left" else "Turn right."
            return AmapInstructionResult(text=text, step_index=current_step_index, kind="turn_now")
        if dist_to_next_turn_m <= DIST_APPROACH_M:
            road = f" onto {step.road}" if step.road else ""
            text = f"In about {int(round(dist_to_next_turn_m))} meters, turn {turn_type}{road}."
            if text != last_spoken or (now_ms - last_spoken_time) >= MIN_SAME_INSTRUCTION_INTERVAL_MS:
                return AmapInstructionResult(text=text, step_index=current_step_index, kind="turn_approach")
        return None
    next_dist = round(dist_to_next_turn_m / 100) * 100
    if next_dist >= DIST_STRAIGHT_M:
        text = f"Continue straight. Next turn in about {int(next_dist)} meters."
        try:
            last_num = float(re.sub(r"\D", "", last_spoken) or "0")
        except Exception:
            last_num = 0
        if text != last_spoken and (
            (now_ms - last_spoken_time) >= MIN_SAME_INSTRUCTION_INTERVAL_MS
            or abs(next_dist - last_num) >= MIN_DISTANCE_DELTA_M
        ):
            return AmapInstructionResult(text=text, step_index=current_step_index, kind="straight")
        return None
    if turn_type in ("left", "right"):
        road = f" onto {step.road}" if step.road else ""
        text = f"In about {int(round(dist_to_next_turn_m))} meters, turn {turn_type}{road}."
        if text != last_spoken or (now_ms - last_spoken_time) >= MIN_SAME_INSTRUCTION_INTERVAL_MS:
            return AmapInstructionResult(text=text, step_index=current_step_index, kind="turn_approach")
    return None


def get_display_info_for_position(
    route: AmapWalkingRoute, position: Tuple[float, float]
) -> Optional[AmapDisplayInfo]:
    steps = route.steps
    if not steps:
        return AmapDisplayInfo(instruction_text="", distance_to_next_m=0, turn_type="straight", road_name="")
    plng, plat = position
    current_step_index = 0
    min_dist = 1e9
    dist_to_next_turn_m = 0.0
    for i, step in enumerate(steps):
        pts = _parse_polyline(step.polyline)
        if not pts:
            continue
        step_end = pts[-1]
        d_end = _haversine_m((plng, plat), step_end)
        if d_end < min_dist:
            min_dist = d_end
            current_step_index = i
            dist_to_next_turn_m = d_end
    step = steps[current_step_index]
    step_dist_m = float(step.distance) if step.distance else 0
    turn_type = _get_turn_type(step.instruction)
    is_last_step = current_step_index >= len(steps) - 1
    if is_last_step and step_dist_m < 20 and min_dist < 25:
        return AmapDisplayInfo(
            instruction_text="You have arrived at your destination",
            distance_to_next_m=0,
            turn_type="arrived",
            road_name=step.road or "",
        )
    road_name = step.road or ""
    turn = "left" if turn_type == "left" else "right" if turn_type == "right" else "cross" if turn_type == "cross" else "straight"
    if turn == "cross":
        if dist_to_next_turn_m <= DIST_TURN_NOW_M:
            return AmapDisplayInfo(
                instruction_text="Cross the road now",
                distance_to_next_m=round(dist_to_next_turn_m),
                turn_type="cross",
                road_name=road_name,
            )
        return AmapDisplayInfo(
            instruction_text=f"In {int(round(dist_to_next_turn_m))} m, cross the road",
            distance_to_next_m=round(dist_to_next_turn_m),
            turn_type="cross",
            road_name=road_name,
        )
    if turn in ("left", "right"):
        if dist_to_next_turn_m <= DIST_TURN_NOW_M:
            return AmapDisplayInfo(
                instruction_text="Turn left" if turn == "left" else "Turn right",
                distance_to_next_m=round(dist_to_next_turn_m),
                turn_type=turn,
                road_name=road_name,
            )
        return AmapDisplayInfo(
            instruction_text=f"In {int(round(dist_to_next_turn_m))} m {'turn left' if turn == 'left' else 'turn right'}{' onto ' + road_name if road_name else ''}",
            distance_to_next_m=round(dist_to_next_turn_m),
            turn_type=turn,
            road_name=road_name,
        )
    display_dist = f"{dist_to_next_turn_m / 1000:.1f} km" if dist_to_next_turn_m >= 1000 else f"{int(round(dist_to_next_turn_m))} m"
    return AmapDisplayInfo(
        instruction_text=f"Continue straight {display_dist}",
        distance_to_next_m=round(dist_to_next_turn_m),
        turn_type="straight",
        road_name=road_name,
    )


async def geocode_address(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    if _navigation_provider == "google":
        return await _google_geocode(address, api_key)
    return await _amap_geocode(address, api_key)


async def _amap_geocode(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    url = f"{AMAP_GEOCODE_URL}?key={api_key}&address={address}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        if data.get("status") != "1" or not data.get("geocodes"):
            logger.warning("[Map API] Amap Geocoding failed or no result: %s", data.get("status"))
            return None
        loc = data["geocodes"][0]["location"]
        lng, lat = map(float, loc.split(","))
        logger.info("[Map API] Amap Geocoding ok: lng=%s, lat=%s", lng, lat)
        return (lng, lat)
    except Exception as e:
        logger.exception("[Map API] Amap Geocoding error: %s", e)
        return None


async def _google_geocode(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    url = f"{GOOGLE_GEOCODE_URL}?address={address}&key={api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            logger.warning("[Map API] Google Geocoding failed: %s", data.get("status"))
            return None
        loc = data["results"][0].get("geometry", {}).get("location")
        if not loc:
            return None
        lng, lat = loc["lng"], loc["lat"]
        logger.info("[Map API] Google Geocoding ok: lng=%s, lat=%s", lng, lat)
        return (lng, lat)
    except Exception as e:
        logger.exception("[Map API] Google Geocoding error: %s", e)
        return None


async def get_walking_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    api_key: str,
) -> Optional[AmapWalkingRoute]:
    if _navigation_provider == "google":
        return await _google_walking_route(origin, destination, api_key)
    return await _amap_walking_route(origin, destination, api_key)


async def _amap_walking_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    api_key: str,
) -> Optional[AmapWalkingRoute]:
    o_str = f"{origin[0]},{origin[1]}"
    d_str = f"{destination[0]},{destination[1]}"
    url = f"{AMAP_WALKING_URL}?key={api_key}&origin={o_str}&destination={d_str}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        if data.get("status") != "1" or not data.get("route", {}).get("paths"):
            logger.warning("[Map API] Amap Directions failed: %s", data.get("status"))
            return None
        path = data["route"]["paths"][0]
        o_parts = data["route"]["origin"].split(",")
        d_parts = data["route"]["destination"].split(",")
        steps = []
        for s in path.get("steps", []):
            steps.append(
                AmapStep(
                    instruction=s.get("instruction", ""),
                    road=s.get("road", ""),
                    distance=str(s.get("distance", 0)),
                    duration=str(s.get("duration", 0)),
                    polyline=s.get("polyline", ""),
                )
            )
        return AmapWalkingRoute(
            origin=(float(o_parts[0]), float(o_parts[1])),
            destination=(float(d_parts[0]), float(d_parts[1])),
            distance=float(path.get("distance", 0)),
            duration=float(path.get("duration", 0)),
            steps=steps,
        )
    except Exception as e:
        logger.exception("[Map API] Amap Directions error: %s", e)
        return None


async def _google_walking_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    api_key: str,
) -> Optional[AmapWalkingRoute]:
    o_str = f"{origin[1]},{origin[0]}"
    d_str = f"{destination[1]},{destination[0]}"
    url = f"{GOOGLE_DIRECTIONS_URL}?origin={o_str}&destination={d_str}&mode=walking&key={api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
        if data.get("status") != "OK" or not data.get("routes"):
            logger.warning("[Map API] Google Directions failed: %s", data.get("status"))
            return None
        leg = data["routes"][0]["legs"][0]
        steps = []
        for s in leg.get("steps", []):
            pts = []
            if s.get("polyline", {}).get("points"):
                pts = _decode_google_polyline(s["polyline"]["points"])
            polyline_str = ";".join(f"{p[0]},{p[1]}" for p in pts)
            instr = re.sub(r"<[^>]+>", "", s.get("html_instructions", "") or "").strip() or (s.get("maneuver") or "Continue")
            steps.append(
                AmapStep(
                    instruction=instr,
                    road="",
                    distance=str(s.get("distance", {}).get("value", 0)),
                    duration=str(s.get("duration", {}).get("value", 0)),
                    polyline=polyline_str,
                )
            )
        return AmapWalkingRoute(
            origin=origin,
            destination=destination,
            distance=float(leg.get("distance", {}).get("value", 0)),
            duration=float(leg.get("duration", {}).get("value", 0)),
            steps=steps,
        )
    except Exception as e:
        logger.exception("[Map API] Google Directions error: %s", e)
        return None
