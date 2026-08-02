"""
weather_tool.py

位置解析优先级：
1. 用户明确传入的位置
2. 前端提供的 GPS 经纬度
3. 客户端 IP 定位
4. 返回需要用户补充位置
"""

from __future__ import annotations

import os
import time
from ipaddress import ip_address
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

import requests
from dotenv import load_dotenv
from langchain.tools import ToolRuntime, tool
from pydantic import Field

from utils.logger_handler import logger


load_dotenv()


XWEATHER_CLIENT_ID = os.getenv("XWEATHER_CLIENT_ID")
XWEATHER_CLIENT_SECRET = os.getenv("XWEATHER_CLIENT_SECRET")
IPWHO_API_KEY = os.getenv("IPWHO_API_KEY")

XWEATHER_BASE_URL = os.getenv(
    "XWEATHER_BASE_URL",
    "https://data.api.xweather.com",
).rstrip("/")
TEXT_GEOCODING_URL = os.getenv(
    "TEXT_GEOCODING_URL",
    "https://geocoding-api.open-meteo.com/v1/search",
)
IPWHO_URL = os.getenv(
    "IPWHO_URL",
    "https://api.ipwho.org",
).rstrip("/")
HTTP_MAX_ATTEMPTS = max(
    1,
    int(os.getenv("WEATHER_HTTP_MAX_ATTEMPTS", "3")),
)
HTTP_BACKOFF_SECONDS = max(
    0.0,
    float(os.getenv("WEATHER_HTTP_BACKOFF_SECONDS", "0.5")),
)


@dataclass
class AgentContext:
    """
    每次调用 Agent 时，由你的 Web 层、App 层或者 CLI 层传入。

    latitude / longitude:
        浏览器或客户端定位得到的坐标。

    client_ip:
        服务端读取到的真实用户 IP。
        不要让大模型生成这个字段。

    user_id:
        由应用认证层传入的当前用户标识，用于隔离长期记忆。
        不要从模型参数或自然语言中获取。

    resolved_location:
        get_location 在本轮工具链中写入的一次性可信定位结果。
        仅供 get_weather 消费，不对模型暴露为工具参数。

    confirmed_location / prefer_confirmed_location:
        天气委派从同一线程恢复的可信地点。仅当用户使用“这里、当地”
        等指代表达时启用，避免模型把上一轮的本地化展示名称重新解析。
    """

    report: bool = False
    latitude: float | None = None
    longitude: float | None = None
    client_ip: str | None = None
    user_id: str | None = None
    resolved_location: dict[str, Any] | None = None
    confirmed_location: dict[str, Any] | None = None
    prefer_confirmed_location: bool = False


_XWEATHER_LOCATION_ALIASES = {
    "东京": "Tokyo,JP",
    "東京": "Tokyo,JP",
    "东京都": "Tokyo,JP",
    "東京都": "Tokyo,JP",
    "日本东京": "Tokyo,JP",
    "日本東京": "Tokyo,JP",
    "日本东京都": "Tokyo,JP",
    "日本東京都": "Tokyo,JP",
    "tokyo": "Tokyo,JP",
}


def _canonical_xweather_location(location: str) -> str:
    return _XWEATHER_LOCATION_ALIASES.get(location.casefold(), location)


def _check_xweather_config() -> None:
    if not XWEATHER_CLIENT_ID or not XWEATHER_CLIENT_SECRET:
        raise RuntimeError(
            "缺少 XWEATHER_CLIENT_ID 或 XWEATHER_CLIENT_SECRET 环境变量"
        )


def _request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """统一处理 HTTP 请求、脱敏日志与瞬时网络错误重试。"""

    service = _service_name(url)

    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={
                    "User-Agent": "agent-weather-service/1.0",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code is not None and status_code >= 500:
                if _retry_request(service, attempt, "HTTPError"):
                    continue

            if service == "IPWho" and status_code in {401, 403}:
                raise RuntimeError(
                    f"IPWho 定位服务拒绝访问（HTTP {status_code}），"
                    "请检查 API key、套餐状态和来源限制"
                ) from exc

            detail = f"HTTP {status_code}" if status_code else "HTTP 请求异常"
            raise RuntimeError(f"{service}调用失败（{detail}）") from exc
        except requests.Timeout as exc:
            if _retry_request(service, attempt, type(exc).__name__):
                continue
            raise RuntimeError(
                f"{service}请求超时（已尝试 {HTTP_MAX_ATTEMPTS} 次）"
            ) from exc
        except requests.exceptions.SSLError as exc:
            if _retry_request(service, attempt, type(exc).__name__):
                continue
            raise RuntimeError(
                f"{service} TLS/SSL 握手失败（已尝试 {HTTP_MAX_ATTEMPTS} 次）"
            ) from exc
        except requests.ConnectionError as exc:
            if _retry_request(service, attempt, type(exc).__name__):
                continue
            raise RuntimeError(
                f"{service}网络连接失败（已尝试 {HTTP_MAX_ATTEMPTS} 次）"
            ) from exc
        except requests.RequestException as exc:
            if _retry_request(service, attempt, type(exc).__name__):
                continue
            raise RuntimeError(
                f"{service}调用失败（网络异常，已尝试 {HTTP_MAX_ATTEMPTS} 次）"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"{service}返回了非 JSON 数据") from exc

    raise RuntimeError(f"{service}调用失败")


def _service_name(url: str) -> str:
    """根据主机名生成不包含路径或认证参数的日志标签。"""

    host = (urlsplit(url).hostname or "").lower()
    if host == (urlsplit(IPWHO_URL).hostname or "").lower():
        return "IPWho"
    if host == (urlsplit(XWEATHER_BASE_URL).hostname or "").lower():
        return "XWeather"
    if host == (urlsplit(TEXT_GEOCODING_URL).hostname or "").lower():
        return "地点解析服务"
    return host or "外部服务"


def _retry_request(service: str, attempt: int, error_type: str) -> bool:
    """记录脱敏的重试信息，并按指数退避等待。"""

    if attempt >= HTTP_MAX_ATTEMPTS:
        logger.error(
            "[外部服务请求失败] service=%s attempts=%s error_type=%s",
            service,
            attempt,
            error_type,
        )
        return False

    delay = HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1))
    logger.warning(
        "[外部服务请求重试] service=%s attempt=%s/%s "
        "error_type=%s retry_in=%.1fs",
        service,
        attempt,
        HTTP_MAX_ATTEMPTS,
        error_type,
        delay,
    )
    time.sleep(delay)
    return True


def _normalize_xweather_response(
    response: Any,
) -> dict[str, Any]:
    """
    XWeather 某些接口返回对象，某些接口返回数组。
    这里统一取第一条结果。
    """

    if isinstance(response, list):
        if not response:
            raise RuntimeError("没有查询到对应位置")
        response = response[0]

    if not isinstance(response, dict):
        raise RuntimeError("XWeather 返回的位置数据格式异常")

    return response


def resolve_text_location(location: str) -> dict[str, Any]:
    """
    使用 XWeather Places 接口标准化用户输入的城市或地区。

    例如：
        广州,广东,中国
        tokyo,japan
        重庆

    返回统一的纬度、经度和行政区信息。
    """

    _check_xweather_config()

    location = location.strip()
    if not location:
        raise ValueError("location 不能为空")

    xweather_location = _canonical_xweather_location(location)
    encoded_location = quote(xweather_location, safe=",+")

    url = f"{XWEATHER_BASE_URL}/places/{encoded_location}"

    data = _request_json(
        url,
        params={
            "client_id": XWEATHER_CLIENT_ID,
            "client_secret": XWEATHER_CLIENT_SECRET,
        },
    )

    if not data.get("success"):
        error = data.get("error") or {}
        error_code = error.get("code") if isinstance(error, dict) else None
        description = (
            error.get("description")
            if isinstance(error, dict)
            else str(error)
        )

        # XWeather 的地点 ID 对中文名称支持不稳定。只有“地点不存在”时
        # 才降级到多语言地理编码；认证、额度等错误必须原样暴露。
        if error_code != "invalid_location":
            raise RuntimeError(
                f"无法识别地点“{location}”：{description or '未知错误'}"
            )

        return _resolve_with_text_geocoder(location)

    result = _normalize_xweather_response(data.get("response"))

    loc = result.get("loc") or {}
    place = result.get("place") or {}
    profile = result.get("profile") or {}

    latitude = loc.get("lat")
    longitude = loc.get("long")

    if latitude is None or longitude is None:
        raise RuntimeError(f"地点“{location}”没有返回有效经纬度")

    resolved = {
        "query": location,
        "source": "explicit_location",
        "latitude": float(latitude),
        "longitude": float(longitude),
        "city": place.get("name"),
        "state": place.get("state"),
        "state_full": place.get("stateFull"),
        "country_code": place.get("country"),
        "country": place.get("countryFull"),
        "timezone": profile.get("tz"),
    }
    if xweather_location != location:
        resolved["matched_query"] = xweather_location
    return resolved


def _resolve_with_text_geocoder(location: str) -> dict[str, Any]:
    """使用支持多语言的地理编码服务把地点名称转换为 WGS84 坐标。"""

    result: dict[str, Any] | None = None
    matched_query: str | None = None

    for candidate in _text_location_candidates(location):
        data = _request_json(
            TEXT_GEOCODING_URL,
            params={
                "name": candidate,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
        )
        results = data.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            result = results[0]
            matched_query = candidate
            break

    if result is None:
        raise RuntimeError(f"无法识别地点“{location}”")

    if not isinstance(result, dict):
        raise RuntimeError("地理编码服务返回的位置数据格式异常")

    latitude = result.get("latitude")
    longitude = result.get("longitude")
    if latitude is None or longitude is None:
        raise RuntimeError(f"地点“{location}”没有返回有效经纬度")

    return {
        "query": location,
        "matched_query": matched_query,
        "source": "text_geocoder",
        "provider": "open-meteo-geocoding",
        "latitude": float(latitude),
        "longitude": float(longitude),
        "city": result.get("name"),
        "state": result.get("admin1"),
        "state_full": result.get("admin1"),
        "country_code": result.get("country_code"),
        "country": result.get("country"),
        "timezone": result.get("timezone"),
    }


def _text_location_candidates(location: str) -> list[str]:
    """生成适合多语言地理编码的中文行政区名称候选。"""

    suffixes = (
        "特别行政区",
        "维吾尔自治区",
        "壮族自治区",
        "回族自治区",
        "自治区",
        "自治州",
        "自治县",
        "地区",
        "新区",
        "市",
        "区",
        "县",
        "盟",
    )

    def strip_suffix(value: str) -> str:
        for suffix in suffixes:
            if value.endswith(suffix) and len(value) > len(suffix):
                return value[: -len(suffix)]
        return value

    normalized = location.strip()
    alias = _canonical_xweather_location(normalized)
    candidates: list[str] = []
    if alias != normalized:
        candidates.extend([alias.split(",", 1)[0], alias])
    candidates.extend([normalized, strip_suffix(normalized)])

    # 直辖市常以“重庆渝中区”这种连续文本出现。先尝试区县，再以
    # 直辖市本身兜底，避免因为区县库缺失导致整次天气查询失败。
    for municipality in ("北京", "上海", "天津", "重庆"):
        prefixes = (f"{municipality}市", municipality)
        for prefix in prefixes:
            if normalized.startswith(prefix) and len(normalized) > len(prefix):
                remainder = normalized[len(prefix) :]
                candidates.extend([remainder, strip_suffix(remainder), municipality])
                break

    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def resolve_ipwho_location(client_ip: str | None) -> dict[str, Any]:
    """
    使用 IPWho 将用户 IP 转换成标准地区和经纬度。

    client_ip=None 时使用 /me 查询请求发起方的公网出口 IP。
    注意：
    - 本地 CLI 调用时，me 通常是当前电脑的公网出口 IP。
    - 部署在服务器后，me 是服务器 IP，不是用户 IP。
    """

    if not IPWHO_API_KEY:
        raise RuntimeError(
            "没有 GPS 坐标，并且未配置 IPWHO_API_KEY，无法进行 IP 定位"
        )

    if client_ip:
        try:
            normalized_ip = str(ip_address(client_ip.strip()))
        except ValueError as exc:
            raise RuntimeError("client_ip 不是有效的 IPv4 或 IPv6 地址") from exc
        encoded_ip = quote(normalized_ip, safe=":")
        url = f"{IPWHO_URL}/ip/{encoded_ip}"
        lookup_type = "client_ip"
    else:
        url = f"{IPWHO_URL}/me"
        lookup_type = "request_ip"

    data = _request_json(
        url,
        params={"apiKey": IPWHO_API_KEY},
    )

    if data.get("success") is not True:
        error = data.get("message") or data.get("error") or "未知错误"
        raise RuntimeError(f"IPWho 定位失败：{error}")

    payload = data.get("data")
    if not isinstance(payload, dict):
        raise RuntimeError("IPWho 返回的数据格式异常：缺少 data 对象")

    geo = payload.get("geoLocation") or payload.get("geo_location") or payload
    if not isinstance(geo, dict):
        raise RuntimeError("IPWho 返回的数据格式异常：缺少地理位置对象")

    def geo_value(*keys: str) -> Any:
        for source in (geo, payload):
            for key in keys:
                if key in source and source[key] is not None:
                    return source[key]
        return None

    latitude = geo_value("latitude")
    longitude = geo_value("longitude")

    if latitude is None or longitude is None:
        raise RuntimeError("IPWho 定位结果中没有有效经纬度")

    city = geo_value("city")
    region = geo_value("region")
    region_code = geo_value("regionCode", "region_code")
    country = geo_value("country")
    country_code = geo_value("countryCode", "country_code")
    timezone_data = payload.get("timezone") or geo.get("timezone") or {}
    timezone = (
        timezone_data.get("time_zone")
        if isinstance(timezone_data, dict)
        else timezone_data
    )
    xweather_location = _standardize_xweather_location(
        city=city,
        region=region,
        region_code=region_code,
        country=country,
        country_code=country_code,
    )

    return {
        "source": "ipwho",
        "lookup_type": lookup_type,
        "ip": payload.get("ip") or data.get("ip"),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "city": city,
        "state": region,
        "state_code": region_code,
        "country": country,
        "country_code": country_code,
        "timezone": timezone,
        "accuracy_radius_km": geo_value("accuracy_radius"),
        "xweather_location": xweather_location,
    }


def _standardize_xweather_location(
    *,
    city: Any,
    region: Any,
    region_code: Any,
    country: Any,
    country_code: Any,
) -> str | None:
    """将 IPWho 行政区字段转换成 XWeather 支持的地点格式。"""

    raw_parts = (city, region_code or region, country_code or country)
    parts: list[str] = []
    seen: set[str] = set()
    for value in raw_parts:
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value.strip()
        identity = normalized.casefold()
        if identity in seen:
            continue
        parts.append(normalized)
        seen.add(identity)
    return ",".join(parts) if parts else None


def resolve_location(
    *,
    explicit_location: str | None,
    context: AgentContext,
) -> dict[str, Any]:
    """
    企业级位置选择逻辑。

    明确地点永远优先于用户当前位置。
    """

    # “这里、当地”等指代由委派层绑定到同一线程中最近一次可信定位。
    # 该开关来自代码对原始用户消息的判断，不接受模型参数控制。
    if context.prefer_confirmed_location:
        confirmed = context.confirmed_location
        if isinstance(confirmed, dict):
            return dict(confirmed)

    # 1. 用户明确指定了地点
    if explicit_location and explicit_location.strip():
        return resolve_text_location(explicit_location)

    return resolve_user_location(context)


def resolve_user_location(context: AgentContext) -> dict[str, Any]:
    """从可信运行时上下文解析当前用户的位置。"""

    # 1. 前端或客户端传入了 GPS 经纬度
    if context.latitude is not None and context.longitude is not None:
        return {
            "source": "device",
            "latitude": float(context.latitude),
            "longitude": float(context.longitude),
            "city": None,
            "state": None,
            "country_code": None,
            "timezone": None,
        }

    # 2. IPWho 定位兜底
    return resolve_ipwho_location(context.client_ip)


def query_xweather_current(
    latitude: float,
    longitude: float,
    *,
    xweather_location: str | None = None,
) -> dict[str, Any]:
    """
    查询坐标处的全球插值当前天气。

    Observations 依赖附近气象站，在部分中国城市会返回 warn_no_data。
    Conditions 使用 XWeather 的全球插值数据，更适合“当前天气”查询。
    """

    _check_xweather_config()

    coordinate_location = f"{latitude:.6f},{longitude:.6f}"
    candidates = [xweather_location, coordinate_location]
    unique_candidates = list(
        dict.fromkeys(candidate.strip() for candidate in candidates if candidate)
    )
    last_error: Any = None

    for candidate in unique_candidates:
        encoded_location = quote(candidate, safe=",.-_")
        url = f"{XWEATHER_BASE_URL}/conditions/{encoded_location}"
        data = _request_json(
            url,
            params={
                "client_id": XWEATHER_CLIENT_ID,
                "client_secret": XWEATHER_CLIENT_SECRET,
            },
        )

        if not data.get("success"):
            last_error = data.get("error") or "未获得天气数据"
            continue

        try:
            result = _normalize_xweather_response(data.get("response"))
        except RuntimeError as exc:
            last_error = str(exc)
            continue

        periods = result.get("periods")
        if not isinstance(periods, list) or not periods:
            last_error = "XWeather Conditions 未返回有效的当前天气数据"
            continue

        current = periods[0]
        if not isinstance(current, dict):
            last_error = "XWeather Conditions 返回的当前天气格式异常"
            continue
        break
    else:
        raise RuntimeError(f"XWeather 查询失败：{last_error}")

    place = result.get("place") or {}
    profile = result.get("profile") or {}
    actual_loc = result.get("loc") or {}

    return {
        "data_source": "xweather_conditions",
        "requested_location": candidate,
        "location": {
            "latitude": actual_loc.get("lat"),
            "longitude": actual_loc.get("long"),
            "city": place.get("city") or place.get("name"),
            "state": place.get("state"),
            "country_code": place.get("country"),
            "timezone": profile.get("tz"),
        },
        "weather": {
            "observation_time": current.get("dateTimeISO"),
            "temperature_c": current.get("tempC"),
            "feels_like_c": current.get("feelslikeC"),
            "humidity": current.get("humidity"),
            "weather": current.get("weather"),
            "weather_primary": current.get("weatherPrimary"),
            "precipitation_mm": current.get("precipMM"),
            "precipitation_probability": current.get("pop"),
            "wind_direction": current.get("windDir"),
            "wind_speed_kph": current.get("windSpeedKPH"),
            "visibility_km": current.get("visibilityKM"),
            "pressure_mb": current.get("pressureMB"),
        },
    }


@tool
def get_location(
    runtime: ToolRuntime[AgentContext],
    location: Annotated[
        str | None,
        Field(
            description=(
                "用户明确指定的查询地点，例如：重庆渝中区、广州、东京。"
                "如果用户只说“这里”“当地”“我这边”，则传入 null，"
                "由工具使用运行时上下文中的用户位置。"
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """
    查询并标准化指定地区或用户当前所在地区。

    使用规则：
    1. 用户明确指定城市、区县或地区时，location 必须传该地点。
    2. 用户只说“这里、当地、我这边”时，location 传 null。
    3. 不允许根据语言、时区或历史对话猜测用户当前位置。
    """

    resolved = resolve_location(
        explicit_location=location,
        context=runtime.context,
    )
    runtime.context.resolved_location = resolved
    return resolved


@tool
def get_weather(
    runtime: ToolRuntime[AgentContext],
) -> dict[str, Any]:
    """
    查询实时天气。

    该工具没有模型可填写的地点参数，只能消费本轮 get_location
    写入的可信定位结果；未先完成定位时调用会直接失败。
    """

    resolved = runtime.context.resolved_location
    if not isinstance(resolved, dict):
        raise RuntimeError(
            "尚未获得可信定位结果：必须先调用 get_location，再调用 get_weather"
        )

    try:
        latitude = float(resolved["latitude"])
        longitude = float(resolved["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        runtime.context.resolved_location = None
        raise RuntimeError("get_location 返回的经纬度无效") from exc

    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        runtime.context.resolved_location = None
        raise RuntimeError("get_location 返回的经纬度超出有效范围")

    # 一次性消费，禁止在没有重新定位的情况下重复查询或沿用陈旧位置。
    runtime.context.resolved_location = None

    weather = query_xweather_current(
        latitude=latitude,
        longitude=longitude,
        xweather_location=resolved.get("xweather_location"),
    )
    return {"resolved_location": resolved, **weather}
