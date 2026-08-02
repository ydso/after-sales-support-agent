"""
agent的中间件开发
"""
import hashlib
from typing import Any, Callable
from utils.prompt_loader import load_report_prompts,load_system_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from agent.memory.long_term import render_user_memories_for_prompt
from utils.logger_handler import logger


def tool_args_for_log(
        tool_name: str,
        tool_args: dict[str, Any],
) -> dict[str, Any]:
    """返回可安全写入日志的工具参数摘要。"""

    if tool_name == "web_search":
        # 搜索词会被发送给外部服务，可能包含用户输入；日志只保留不可逆摘要。
        query = str(tool_args.get("query") or "")
        return {
            "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:12],
            "query_length": len(query),
        }
    return tool_args


#工具执行的监控
@wrap_tool_call
def monitor_tool(
        #请求的数据封装
        request: ToolCallRequest,
        #执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:

    tool_name = request.tool_call["name"]
    safe_args = tool_args_for_log(tool_name, request.tool_call["args"])
    logger.info("[monitor_tool]执行工具%s", tool_name)
    logger.info("[monitor_tool]传入参数摘要%s", safe_args)

    try:
        result = handler(request)
        logger.info(f"[monitor_tool]工具{request.tool_call['name']}调用成功")
        if request.tool_call['name'] == "fill_context_for_report":
            context = request.runtime.context
            if isinstance(context, dict):
                context["report"] = True
            else:
                context.report = True

        return result
    except Exception as e:
        logger.error(f"[monitor_tool]工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e
#在模型执行前输出日志
@before_model
def log_before_model(
        #整个agent智能体的状态记录
        state:AgentState,
        #记录了执行过程中整个上下文信息
        runtime:Runtime,
):
    logger.info(f"[log_before_model]即将调用模型，并带有{len(state['messages'])}条消息")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None

#动态切换提示词，在检测到用户想要输出报告时，切换对应的提示词，事半功倍
@dynamic_prompt     #每一次在生成提示词之前,调用此函数
def report_prompt_switch(
        request: ModelRequest,
):
    context = request.runtime.context
    is_report = (
        context.get("report", False)
        if isinstance(context, dict)
        else context.report
    )
    if is_report:   #是报告生成场景，返回报告生成提示词
        logger.info(f"[report_prompt_switch]提示词已切换为生成报告提示词")
        base_prompt = load_report_prompts()
    else:
        base_prompt = load_system_prompts()

    user_id = (
        context.get("user_id")
        if isinstance(context, dict)
        else context.user_id
    )
    memory_prompt = render_user_memories_for_prompt(
        request.runtime.store,
        user_id,
    )
    if not memory_prompt:
        return base_prompt

    return f"{base_prompt}\n\n{'=' * 50}\n长期记忆上下文\n{'=' * 50}\n{memory_prompt}"
