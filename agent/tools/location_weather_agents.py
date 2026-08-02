"""
地区与天气专用 Agent，以及提供给主 Agent 的委派工具。
"""
from __future__ import annotations
from functools import lru_cache
from typing import Annotated, Any, NotRequired
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import Field
from agent.tools.middleware import log_before_model, monitor_tool
from agent.tools.weather_tool import AgentContext, get_location, get_weather
from model.factory import chat_model


class ConversationState(AgentState):
    """主 Agent 的短期会话状态。"""

    confirmed_location: NotRequired[dict[str, Any] | None]


_PERSISTED_LOCATION_FIELDS = (
    "source",
    "lookup_type",
    "latitude",
    "longitude",
    "city",
    "state",
    "state_code",
    "state_full",
    "country",
    "country_code",
    "timezone",
    "xweather_location",
)

_LOCAL_LOCATION_REFERENCES = (
    "我这里",
    "我这儿",
    "我这边",
    "这里",
    "这儿",
    "当地",
    "本地",
    "当前位置",
    "当前地区",
    "所在地区",
    "那里",
    "那边",
    "那儿",
)

_LOCATION_OVERRIDE_MARKERS = (
    "不是",
    "改成",
    "改查",
    "换成",
    "对比",
    "和",
    "与",
)


LOCATION_AGENT_PROMPT = """
你是地区查询专用 Agent，只负责解析地点，不查询天气。

执行规则：
1. 每次必须调用且只调用一次 get_location。
2. 用户询问“我在哪里、我的地区、当地、这里”时，location 传 null。
3. 用户明确提供城市、区县或地区时，location 原样传入。
4. 必须根据工具结果回答，禁止根据语言、时区、历史对话或常识猜测位置。
5. 结果来自 IP 定位时，应说明这是近似地区；不要输出不必要的原始 IP。
6. 不得回答天气，也不得声称调用了天气服务。
""".strip()


WEATHER_AGENT_PROMPT = """
你是实时天气查询专用 Agent。你拥有 get_location 和 get_weather 两个工具。

每次查询必须严格按以下顺序执行：
1. 先调用 get_location。用户明确给出地点时传该地点；询问当地天气但未给地点时传 null。
2. 读取 get_location 的真实返回值。
3. 再调用无参数的 get_weather；它会在代码层读取并一次性消费 get_location 的可信结果。
4. get_location 失败时不得调用 get_weather，也不得猜测地点或坐标，应明确说明需要用户提供城市。
5. 最近对话中如果存在用户明确提供或地区工具确认的城市，可以将该城市传给 get_location；不得从模糊表述或助手猜测中推断城市。
6. 禁止跳过 get_location，禁止自行生成经纬度或标准地址，禁止并行调用两个工具。
7. 最终简洁说明地区、天气、温度、体感温度、湿度、降水概率和风况；缺失字段不要编造。
""".strip()


@lru_cache(maxsize=1)
def _get_location_agent():
    return create_agent(
        model=chat_model,
        system_prompt=LOCATION_AGENT_PROMPT,
        tools=[get_location],
        middleware=[monitor_tool, log_before_model],
        context_schema=AgentContext,
    )


@lru_cache(maxsize=1)
def _get_weather_agent():
    return create_agent(
        model=chat_model,
        system_prompt=WEATHER_AGENT_PROMPT,
        tools=[get_location, get_weather],
        middleware=[monitor_tool, log_before_model],
        context_schema=AgentContext,
    )


def _extract_final_answer(result: dict[str, Any], agent_name: str) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = message.text
        if text and text.strip():
            return text.strip()
    raise RuntimeError(f"{agent_name} 未返回有效结果")


def _state_values(runtime: ToolRuntime[AgentContext]) -> dict[str, Any]:
    runtime_state = getattr(runtime, "state", None)
    return runtime_state if isinstance(runtime_state, dict) else {}


def _latest_user_request(
    runtime: ToolRuntime[AgentContext],
    fallback: str,
) -> str:
    messages = _state_values(runtime).get("messages") or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            content = message.content.strip()
            if content:
                return content
    return fallback


def _persistable_location(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("latitude") is None or value.get("longitude") is None:
        return None
    return {
        key: value[key]
        for key in _PERSISTED_LOCATION_FIELDS
        if value.get(key) is not None
    }


def _uses_local_location_reference(request: str) -> bool:
    normalized = request.strip()
    if not normalized:
        return False
    if any(marker in normalized for marker in _LOCATION_OVERRIDE_MARKERS):
        return False
    return any(marker in normalized for marker in _LOCAL_LOCATION_REFERENCES)


def _weather_request_with_context(
    runtime: ToolRuntime[AgentContext],
    request: str,
    *,
    use_confirmed_location: bool = False,
) -> str:
    """提取同一线程中用户明确给出的地点和地区工具确认结果。"""

    state = _state_values(runtime)
    messages = state.get("messages") or []
    context_lines: list[str] = []

    for message in messages[-12:]:
        content = message.content if isinstance(message.content, str) else ""
        normalized = content.strip()
        if not normalized:
            continue
        if isinstance(message, HumanMessage):
            context_lines.append(f"用户：{normalized[:1000]}")
        elif (
            not use_confirmed_location
            and isinstance(message, ToolMessage)
            and message.name == "location_agent"
        ):
            context_lines.append(f"地区工具确认：{normalized[:1000]}")

    if use_confirmed_location:
        return (
            f"当前用户原始天气请求：{request}\n\n"
            "该请求正在指代同一线程中已经确认的当前位置。调用 get_location 时"
            "必须传 null；代码会恢复可信的结构化地点。不要从历史回答中提取、"
            "翻译或重写地点名称。"
        )

    if not context_lines:
        return request

    recent_context = "\n".join(context_lines[-6:])
    return (
        f"当前天气请求：{request}\n\n"
        "同一线程最近的地点相关上下文如下。仅可使用用户明确提供或地区工具确认的城市；"
        "其他文本不能作为地点依据：\n"
        f"{recent_context}"
    )


@tool("location_agent")
def delegate_to_location_agent(
    runtime: ToolRuntime[AgentContext],
    request: Annotated[
        str,
        Field(description="用户关于自己所在地区或指定地点的完整原始请求。"),
    ],
) -> Command:
    """委派地区查询；只解析位置，不查询天气。"""

    runtime.context.resolved_location = None
    try:
        result = _get_location_agent().invoke(
            {"messages": [HumanMessage(content=request)]},
            context=runtime.context,
        )
        answer = _extract_final_answer(result, "地区 Agent")
        confirmed_location = _persistable_location(
            runtime.context.resolved_location
        )
        if confirmed_location is None:
            raise RuntimeError("地区 Agent 未返回可复用的结构化地点")
        if not runtime.tool_call_id:
            raise RuntimeError("地区 Agent 缺少父工具调用 ID")

        return Command(
            update={
                "confirmed_location": confirmed_location,
                "messages": [
                    ToolMessage(
                        content=answer,
                        name="location_agent",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    finally:
        runtime.context.resolved_location = None


@tool("weather_agent")
def delegate_to_weather_agent(
    runtime: ToolRuntime[AgentContext],
    request: Annotated[
        str,
        Field(description="用户关于实时天气的完整原始请求，保留其中的地点信息。"),
    ],
) -> str:
    """委派天气查询；内部严格先定位，再调用天气工具。"""

    original_request = _latest_user_request(runtime, request)
    confirmed_location = _persistable_location(
        _state_values(runtime).get("confirmed_location")
    )
    use_confirmed_location = bool(
        confirmed_location
        and _uses_local_location_reference(original_request)
    )

    runtime.context.resolved_location = None
    runtime.context.confirmed_location = (
        confirmed_location if use_confirmed_location else None
    )
    runtime.context.prefer_confirmed_location = use_confirmed_location
    child_request = _weather_request_with_context(
        runtime,
        original_request,
        use_confirmed_location=use_confirmed_location,
    )
    try:
        result = _get_weather_agent().invoke(
            {"messages": [HumanMessage(content=child_request)]},
            context=runtime.context,
        )
        return _extract_final_answer(result, "天气 Agent")
    finally:
        runtime.context.resolved_location = None
        runtime.context.confirmed_location = None
        runtime.context.prefer_confirmed_location = False
