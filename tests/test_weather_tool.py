from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent.tools import weather_tool


class WeatherToolTests(unittest.TestCase):
    @patch.object(weather_tool, "_request_json")
    def test_tokyo_prefecture_alias_uses_xweather_canonical_name(
        self,
        request_json,
    ) -> None:
        request_json.return_value = {
            "success": True,
            "response": [
                {
                    "loc": {"lat": 35.6895, "long": 139.69171},
                    "place": {
                        "name": "Tokyo",
                        "state": "Tokyo",
                        "country": "JP",
                    },
                    "profile": {"tz": "Asia/Tokyo"},
                }
            ],
        }

        with (
            patch.object(weather_tool, "XWEATHER_CLIENT_ID", "client-id"),
            patch.object(weather_tool, "XWEATHER_CLIENT_SECRET", "secret"),
        ):
            for location in ("东京都", "東京都", "东京", "東京", "Tokyo"):
                with self.subTest(location=location):
                    result = weather_tool.resolve_text_location(location)
                    self.assertIn(
                        "/places/Tokyo,JP",
                        request_json.call_args.args[0],
                    )
                    self.assertEqual(result["query"], location)
                    self.assertEqual(result["matched_query"], "Tokyo,JP")

    @patch.object(weather_tool, "resolve_text_location")
    def test_confirmed_conversation_location_wins_for_local_reference(
        self,
        resolve_text_location,
    ) -> None:
        confirmed_location = {
            "source": "ipwho",
            "latitude": 35.6893,
            "longitude": 139.6899,
            "city": "Tokyo",
            "country_code": "JP",
            "xweather_location": "Tokyo,13,JP",
        }
        context = weather_tool.AgentContext(
            confirmed_location=confirmed_location,
            prefer_confirmed_location=True,
        )

        result = weather_tool.resolve_location(
            explicit_location="东京都",
            context=context,
        )

        self.assertEqual(result, confirmed_location)
        self.assertIsNot(result, confirmed_location)
        resolve_text_location.assert_not_called()

    @patch.object(weather_tool, "_request_json")
    def test_ipwho_specific_ip_builds_standard_xweather_location(
        self,
        request_json,
    ) -> None:
        request_json.return_value = {
            "success": True,
            "data": {
                "ip": "203.0.113.10",
                "geoLocation": {
                    "country": "China",
                    "countryCode": "CN",
                    "region": "Chongqing",
                    "regionCode": "CQ",
                    "city": "Chongqing",
                    "latitude": 29.56,
                    "longitude": 106.55,
                    "accuracy_radius": 20,
                },
                "timezone": {"time_zone": "Asia/Shanghai"},
            },
        }

        with (
            patch.object(weather_tool, "IPWHO_API_KEY", "valid-key"),
            patch.object(weather_tool, "IPWHO_URL", "https://api.ipwho.org"),
        ):
            result = weather_tool.resolve_ipwho_location("203.0.113.10")

        self.assertEqual(result["source"], "ipwho")
        self.assertEqual(result["xweather_location"], "Chongqing,CQ,CN")
        self.assertEqual(result["timezone"], "Asia/Shanghai")
        self.assertEqual(
            request_json.call_args.args[0],
            "https://api.ipwho.org/ip/203.0.113.10",
        )
        self.assertEqual(request_json.call_args.kwargs["params"], {"apiKey": "valid-key"})

    @patch.object(weather_tool, "_request_json")
    def test_ipwho_self_lookup_accepts_snake_case_response(
        self,
        request_json,
    ) -> None:
        request_json.return_value = {
            "success": True,
            "data": {
                "ip": "198.51.100.20",
                "geo_location": {
                    "country": "China",
                    "country_code": "CN",
                    "region": "Guangdong",
                    "region_code": "GD",
                    "city": "Guangzhou",
                    "latitude": 23.13,
                    "longitude": 113.26,
                },
            },
        }

        with (
            patch.object(weather_tool, "IPWHO_API_KEY", "valid-key"),
            patch.object(weather_tool, "IPWHO_URL", "https://api.ipwho.org"),
        ):
            result = weather_tool.resolve_ipwho_location(None)

        self.assertEqual(result["xweather_location"], "Guangzhou,GD,CN")
        self.assertEqual(
            request_json.call_args.args[0],
            "https://api.ipwho.org/me",
        )

    @patch.object(weather_tool, "_request_json")
    def test_xweather_uses_standard_location_before_coordinates(
        self,
        request_json,
    ) -> None:
        request_json.return_value = {
            "success": True,
            "response": [
                {
                    "loc": {"lat": 29.56, "long": 106.55},
                    "place": {"name": "chongqing", "country": "cn"},
                    "profile": {"tz": "Asia/Shanghai"},
                    "periods": [{"tempC": 32.0, "humidity": 60}],
                }
            ],
        }

        weather_tool.query_xweather_current(
            29.56,
            106.55,
            xweather_location="Chongqing,CQ,CN",
        )

        self.assertIn(
            "/conditions/Chongqing,CQ,CN",
            request_json.call_args.args[0],
        )

    @patch.object(weather_tool, "_request_json")
    def test_xweather_falls_back_to_coordinates(self, request_json) -> None:
        request_json.side_effect = [
            {
                "success": False,
                "error": {"code": "invalid_location"},
            },
            {
                "success": True,
                "response": [
                    {
                        "loc": {"lat": 29.56, "long": 106.55},
                        "place": {"name": "chongqing", "country": "cn"},
                        "profile": {"tz": "Asia/Shanghai"},
                        "periods": [{"tempC": 32.0}],
                    }
                ],
            },
        ]

        result = weather_tool.query_xweather_current(
            29.56,
            106.55,
            xweather_location="Unknown,XX,CN",
        )

        self.assertEqual(request_json.call_count, 2)
        self.assertIn(
            "/conditions/29.560000,106.550000",
            request_json.call_args_list[1].args[0],
        )
        self.assertEqual(result["weather"]["temperature_c"], 32.0)

    @patch.object(weather_tool, "resolve_location")
    def test_get_location_receives_agent_runtime_context(
        self,
        resolve_location,
    ) -> None:
        resolve_location.return_value = {
            "source": "ipwho",
            "latitude": 29.56,
            "longitude": 106.55,
            "xweather_location": "Chongqing,CQ,CN",
        }

        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode([weather_tool.get_location]))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        context = weather_tool.AgentContext(client_ip="203.0.113.10")

        graph.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_location",
                                "args": {"location": None},
                                "id": "location-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            context=context,
        )

        self.assertIs(resolve_location.call_args.kwargs["context"], context)
        self.assertEqual(
            context.resolved_location["xweather_location"],
            "Chongqing,CQ,CN",
        )

    @patch.object(weather_tool, "query_xweather_current")
    def test_get_weather_only_queries_resolved_location(
        self,
        query_xweather_current,
    ) -> None:
        query_xweather_current.return_value = {
            "weather": {"temperature_c": 32.0},
        }
        context = weather_tool.AgentContext(
            resolved_location={
                "source": "ipwho",
                "latitude": 29.56,
                "longitude": 106.55,
                "xweather_location": "Chongqing,CQ,CN",
            }
        )

        result = weather_tool.get_weather.func(
            runtime=SimpleNamespace(context=context),
        )

        self.assertEqual(result["weather"]["temperature_c"], 32.0)
        self.assertEqual(result["resolved_location"]["source"], "ipwho")
        self.assertIsNone(context.resolved_location)
        self.assertEqual(weather_tool.get_weather.args, {})
        query_xweather_current.assert_called_once_with(
            latitude=29.56,
            longitude=106.55,
            xweather_location="Chongqing,CQ,CN",
        )

    def test_get_weather_rejects_calls_without_prior_location(self) -> None:
        context = weather_tool.AgentContext()

        with self.assertRaisesRegex(RuntimeError, "必须先调用 get_location"):
            weather_tool.get_weather.func(
                runtime=SimpleNamespace(context=context),
            )


if __name__ == "__main__":
    unittest.main()
