from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agent.tools import location_weather_agents
from agent.tools import weather_tool
from agent.tools.weather_tool import AgentContext


class LocationWeatherAgentTests(unittest.TestCase):
    def tearDown(self) -> None:
        location_weather_agents._get_location_agent.cache_clear()
        location_weather_agents._get_weather_agent.cache_clear()

    @patch.object(location_weather_agents, "create_agent")
    def test_specialized_agents_have_separate_tool_boundaries(
        self,
        create_agent,
    ) -> None:
        create_agent.side_effect = [Mock(), Mock()]

        location_weather_agents._get_location_agent()
        location_weather_agents._get_weather_agent()

        location_tools = create_agent.call_args_list[0].kwargs["tools"]
        weather_tools = create_agent.call_args_list[1].kwargs["tools"]
        self.assertEqual([tool.name for tool in location_tools], ["get_location"])
        self.assertEqual(
            [tool.name for tool in weather_tools],
            ["get_location", "get_weather"],
        )

    @patch.object(location_weather_agents, "_get_location_agent")
    def test_location_delegation_forwards_runtime_context(
        self,
        get_location_agent,
    ) -> None:
        child_agent = Mock()
        def invoke_child(*_args, context):
            context.resolved_location = {
                "source": "ipwho",
                "ip": "203.0.113.10",
                "latitude": 29.56,
                "longitude": 106.55,
                "city": "Chongqing",
                "state": "Chongqing",
                "country_code": "CN",
                "xweather_location": "Chongqing,CQ,CN",
            }
            return {
                "messages": [AIMessage(content="你当前位于重庆市。")],
            }

        child_agent.invoke.side_effect = invoke_child
        get_location_agent.return_value = child_agent
        context = AgentContext(client_ip="203.0.113.10")

        result = location_weather_agents.delegate_to_location_agent.func(
            runtime=SimpleNamespace(
                context=context,
                tool_call_id="location-agent-call",
            ),
            request="我目前在哪个地区？",
        )

        self.assertIsInstance(result, Command)
        self.assertEqual(
            result.update["confirmed_location"]["xweather_location"],
            "Chongqing,CQ,CN",
        )
        self.assertNotIn("ip", result.update["confirmed_location"])
        tool_message = result.update["messages"][0]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertEqual(tool_message.content, "你当前位于重庆市。")
        self.assertEqual(tool_message.tool_call_id, "location-agent-call")
        self.assertIs(child_agent.invoke.call_args.kwargs["context"], context)
        self.assertIsNone(context.resolved_location)

    @patch.object(location_weather_agents, "_get_location_agent")
    def test_location_delegation_persists_confirmed_location_in_graph_state(
        self,
        get_location_agent,
    ) -> None:
        child_agent = Mock()

        def invoke_child(*_args, context):
            context.resolved_location = {
                "source": "ipwho",
                "ip": "203.0.113.10",
                "latitude": 35.6893,
                "longitude": 139.6899,
                "city": "Tokyo",
                "state": "Tokyo",
                "country_code": "JP",
                "xweather_location": "Tokyo,13,JP",
            }
            return {
                "messages": [AIMessage(content="你当前位于日本东京都。")],
            }

        child_agent.invoke.side_effect = invoke_child
        get_location_agent.return_value = child_agent
        builder = StateGraph(location_weather_agents.ConversationState)
        builder.add_node(
            "tools",
            ToolNode([location_weather_agents.delegate_to_location_agent]),
        )
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()

        result = graph.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "location_agent",
                                "args": {"request": "我在哪里？"},
                                "id": "location-agent-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            context=AgentContext(client_ip="203.0.113.10"),
        )

        self.assertEqual(
            result["confirmed_location"]["xweather_location"],
            "Tokyo,13,JP",
        )
        self.assertNotIn("ip", result["confirmed_location"])
        self.assertEqual(result["messages"][-1].content, "你当前位于日本东京都。")

    @patch.object(location_weather_agents, "_get_weather_agent")
    def test_weather_delegation_forwards_runtime_context(
        self,
        get_weather_agent,
    ) -> None:
        child_agent = Mock()
        child_agent.invoke.return_value = {
            "messages": [AIMessage(content="重庆当前气温 32℃。")],
        }
        get_weather_agent.return_value = child_agent
        context = AgentContext(
            latitude=29.56,
            longitude=106.55,
            resolved_location={"city": "stale-city"},
        )
        runtime = SimpleNamespace(
            context=context,
            state={
                "messages": [
                    HumanMessage(content="这次对话我指定的城市是广州。"),
                    HumanMessage(content="那里的天气怎么样？"),
                ]
            },
        )

        result = location_weather_agents.delegate_to_weather_agent.func(
            runtime=runtime,
            request="那里的天气怎么样？",
        )

        self.assertEqual(result, "重庆当前气温 32℃。")
        self.assertIs(child_agent.invoke.call_args.kwargs["context"], context)
        child_message = child_agent.invoke.call_args.args[0]["messages"][0]
        self.assertIn("广州", child_message.content)
        self.assertIn("那里的天气怎么样", child_message.content)
        self.assertIsNone(context.resolved_location)

    @patch.object(location_weather_agents, "_get_weather_agent")
    def test_local_weather_reuses_confirmed_location_from_thread_state(
        self,
        get_weather_agent,
    ) -> None:
        confirmed_location = {
            "source": "ipwho",
            "latitude": 35.6893,
            "longitude": 139.6899,
            "city": "Tokyo",
            "state": "Tokyo",
            "country_code": "JP",
            "xweather_location": "Tokyo,13,JP",
        }
        observed_context = {}
        child_agent = Mock()

        def invoke_child(*_args, context):
            observed_context["confirmed_location"] = context.confirmed_location
            observed_context["prefer_confirmed_location"] = (
                context.prefer_confirmed_location
            )
            location = location_weather_agents.get_location.func(
                runtime=SimpleNamespace(context=context),
                location="东京都",
            )
            location_weather_agents.get_weather.func(
                runtime=SimpleNamespace(context=context),
            )
            observed_context["resolved_city"] = location["city"]
            return {
                "messages": [AIMessage(content="东京当前天气晴朗。")],
            }

        child_agent.invoke.side_effect = invoke_child
        get_weather_agent.return_value = child_agent
        context = AgentContext()
        runtime = SimpleNamespace(
            context=context,
            state={
                "confirmed_location": confirmed_location,
                "messages": [HumanMessage(content="我这里的天气如何？")],
            },
        )

        with (
            patch.object(weather_tool, "resolve_text_location") as resolve_text,
            patch.object(
                weather_tool,
                "query_xweather_current",
                return_value={"weather": {"temperature_c": 35.0}},
            ) as query_weather,
        ):
            result = location_weather_agents.delegate_to_weather_agent.func(
                runtime=runtime,
                request="我这里的天气如何，当前所在地区日本东京都",
            )

        self.assertEqual(result, "东京当前天气晴朗。")
        self.assertEqual(
            observed_context["confirmed_location"],
            confirmed_location,
        )
        self.assertTrue(observed_context["prefer_confirmed_location"])
        self.assertEqual(observed_context["resolved_city"], "Tokyo")
        resolve_text.assert_not_called()
        query_weather.assert_called_once_with(
            latitude=35.6893,
            longitude=139.6899,
            xweather_location="Tokyo,13,JP",
        )
        child_message = child_agent.invoke.call_args.args[0]["messages"][0]
        self.assertIn("我这里的天气如何", child_message.content)
        self.assertIsNone(context.confirmed_location)
        self.assertFalse(context.prefer_confirmed_location)


if __name__ == "__main__":
    unittest.main()
