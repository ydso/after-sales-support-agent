"""
Xweather 天气 API 客户端。

负责：
1. 调用 Xweather 实时观测接口；
2. 调用最近一小时预报接口；
3. 校验 API 响应；
4. 返回结构化天气数据。

不要在此处编写扫地机器人使用建议。
天气建议应由 Agent 根据工具结果综合判断。
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from utils.logger_handler import logger


load_dotenv()


XWEATHER_BASE_URL = "https://data.api.xweather.com"

OBSERVATION_FIELDS = ",".join(
    [
        "place.name",
        "place.city",
        "place.state",
        "place.country",
        "profile.tz",
        "ob.dateTimeISO",
        "ob.tempC",
        "ob.feelslikeC",
        "ob.dewpointC",
        "ob.humidity",
        "ob.precipMM",
        "ob.windSpeedKPH",
        "ob.windDir",
        "ob.visibilityKM",
        "ob.weather",
        "ob.weatherPrimary",
    ]
)

FORECAST_FIELDS = ",".join(
    [
        "place.name",
        "place.state",
        "place.country",
        "periods.dateTimeISO",
        "periods.pop",
        "periods.precipMM",
        "periods.humidity",
        "periods.weatherPrimary",
    ]
)


class XWeatherAPIError(RuntimeError):
    """Xweather API 调用异常。"""


class XWeatherClient:
    """Xweather 天气服务客户端。"""

    def __init__(self) -> None:
        self.client_id = os.getenv("XWEATHER_CLIENT_ID")
        self.client_secret = os.getenv("XWEATHER_CLIENT_SECRET")

        if not self.client_id:
            raise RuntimeError(
                "没有读取到 XWEATHER_CLIENT_ID，"
                "请检查项目根目录中的 .env 文件。"
            )

        if not self.client_secret:
            raise RuntimeError(
                "没有读取到 XWEATHER_CLIENT_SECRET，"
                "请检查项目根目录中的 .env 文件。"
            )

        self.timeout = httpx.Timeout(
            timeout=12.0,
            connect=5.0,
        )

    def _request(
        self,
        client: httpx.Client,
        endpoint: str,
        location: str,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], httpx.Headers]:
        """
        调用 Xweather API，并统一处理错误。

        对网络异常和服务端 5xx 错误最多重试一次。
        认证失败、参数错误和额度不足不重试。
        """

        encoded_location = quote(
            location,
            safe=",.-_",
        )

        url = (
            f"{XWEATHER_BASE_URL}/"
            f"{endpoint}/{encoded_location}"
        )

        request_params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            **params,
        }

        last_exception: Exception | None = None

        for attempt in range(2):
            try:
                response = client.get(
                    url,
                    params=request_params,
                )

                try:
                    response_data = response.json()
                except ValueError as exc:
                    raise XWeatherAPIError(
                        "天气服务返回了无法解析的数据。"
                    ) from exc

                if response.status_code == 401:
                    raise XWeatherAPIError(
                        "Xweather 认证失败，请检查 "
                        "Client ID、Client Secret 和应用 namespace。"
                    )

                if response.status_code == 429:
                    raise XWeatherAPIError(
                        "Xweather API 调用额度或频率已经达到上限。"
                    )

                if response.status_code >= 500:
                    if attempt == 0:
                        time.sleep(0.5)
                        continue

                    raise XWeatherAPIError(
                        "Xweather 天气服务暂时异常，请稍后重试。"
                    )

                if response.status_code >= 400:
                    error_data = response_data.get("error") or {}

                    error_code = error_data.get(
                        "code",
                        "unknown_error",
                    )
                    description = error_data.get(
                        "description",
                        "请求处理失败",
                    )

                    raise XWeatherAPIError(
                        f"{error_code}：{description}"
                    )

                success = response_data.get("success")

                if success is not True:
                    error_data = response_data.get("error") or {}

                    error_code = error_data.get(
                        "code",
                        "unknown_error",
                    )
                    description = error_data.get(
                        "description",
                        "未获得有效天气数据",
                    )

                    raise XWeatherAPIError(
                        f"{error_code}：{description}"
                    )

                # success=True 时仍可能携带 warning。
                warning = response_data.get("error")

                if warning:
                    logger.warning(
                        "[Xweather警告] endpoint=%s "
                        "location=%s code=%s description=%s",
                        endpoint,
                        location,
                        warning.get("code"),
                        warning.get("description"),
                    )

                return response_data, response.headers

            except XWeatherAPIError:
                raise

            except httpx.TimeoutException as exc:
                last_exception = exc

                if attempt == 0:
                    time.sleep(0.5)
                    continue

            except httpx.RequestError as exc:
                last_exception = exc

                if attempt == 0:
                    time.sleep(0.5)
                    continue

        logger.exception(
            "[Xweather网络请求失败] endpoint=%s location=%s",
            endpoint,
            location,
            exc_info=last_exception,
        )

        raise XWeatherAPIError(
            "当前无法连接 Xweather 天气服务。"
        )

    @staticmethod
    def _extract_observation(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        提取 observation 数据。

        使用 location id 请求时，response 通常是一个对象；
        为兼容其他情况，也支持列表形式。
        """

        response_data = data.get("response")

        if isinstance(response_data, dict):
            return response_data

        if (
            isinstance(response_data, list)
            and response_data
            and isinstance(response_data[0], dict)
        ):
            return response_data[0]

        raise XWeatherAPIError(
            "Xweather 未返回有效的实时观测数据。"
        )

    @staticmethod
    def _extract_first_forecast_period(
        data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        提取最近一个小时预报时段。

        forecasts 接口的 response 通常是数组。
        """

        response_data = data.get("response")

        if not isinstance(response_data, list):
            return None

        if not response_data:
            return None

        forecast_data = response_data[0]

        if not isinstance(forecast_data, dict):
            return None

        periods = forecast_data.get("periods")

        if not isinstance(periods, list) or not periods:
            return None

        first_period = periods[0]

        if not isinstance(first_period, dict):
            return None

        return first_period

    def get_current_weather(
        self,
        city: str,
    ) -> dict[str, Any]:
        """
        查询指定城市的实时观测和最近一小时预报。

        Args:
            city:
                城市名称。为减少重名，推荐附加国家代码，
                例如：
                北京,CN
                Tokyo,JP
                Shanghai,CN

        Returns:
            结构化天气数据。
        """

        normalized_city = city.strip()

        if not normalized_city:
            raise ValueError("城市名称不能为空。")

        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            observation_data, observation_headers = (
                self._request(
                    client=client,
                    endpoint="observations",
                    location=normalized_city,
                    params={
                        "fields": OBSERVATION_FIELDS,
                    },
                )
            )

            observation = self._extract_observation(
                observation_data
            )

            # 小时预报不是实时天气的必要条件。
            # 如果预报查询失败，仍然保留实时观测结果。
            forecast_period: dict[str, Any] | None = None
            forecast_cost_tokens: str | None = None

            try:
                forecast_data, forecast_headers = (
                    self._request(
                        client=client,
                        endpoint="forecasts",
                        location=normalized_city,
                        params={
                            "filter": "1hr",
                            "plimit": 1,
                            "fields": FORECAST_FIELDS,
                        },
                    )
                )

                forecast_period = (
                    self._extract_first_forecast_period(
                        forecast_data
                    )
                )

                forecast_cost_tokens = (
                    forecast_headers.get("X-Cost-Tokens")
                )

            except XWeatherAPIError as exc:
                logger.warning(
                    "[Xweather小时预报失败] city=%s error=%s",
                    normalized_city,
                    exc,
                )

        observation_cost_tokens = (
            observation_headers.get("X-Cost-Tokens")
        )

        logger.info(
            "[Xweather查询成功] city=%s "
            "observation_cost=%s forecast_cost=%s",
            normalized_city,
            observation_cost_tokens,
            forecast_cost_tokens,
        )

        place = observation.get("place") or {}
        weather = observation.get("ob") or {}
        profile = observation.get("profile") or {}

        return {
            "query_city": normalized_city,
            "location": {
                "name": place.get("name"),
                "city": place.get("city"),
                "state": place.get("state"),
                "country": place.get("country"),
                "timezone": profile.get("tz"),
            },
            "current": {
                "observed_at": weather.get("dateTimeISO"),
                "weather": (
                    weather.get("weatherPrimary")
                    or weather.get("weather")
                ),
                "temperature_c": weather.get("tempC"),
                "feels_like_c": weather.get("feelslikeC"),
                "dew_point_c": weather.get("dewpointC"),
                "humidity_percent": weather.get("humidity"),
                "precipitation_mm": weather.get("precipMM"),
                "wind_speed_kph": weather.get(
                    "windSpeedKPH"
                ),
                "wind_direction": weather.get("windDir"),
                "visibility_km": weather.get(
                    "visibilityKM"
                ),
            },
            "nearest_hour_forecast": {
                "forecast_at": (
                    forecast_period.get("dateTimeISO")
                    if forecast_period
                    else None
                ),
                "weather": (
                    forecast_period.get("weatherPrimary")
                    if forecast_period
                    else None
                ),
                "precipitation_probability_percent": (
                    forecast_period.get("pop")
                    if forecast_period
                    else None
                ),
                "expected_precipitation_mm": (
                    forecast_period.get("precipMM")
                    if forecast_period
                    else None
                ),
                "humidity_percent": (
                    forecast_period.get("humidity")
                    if forecast_period
                    else None
                ),
            },
        }