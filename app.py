from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Iterator
from itertools import chain as chain_streams
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from agent.react_agent import ReactAgent


# ============================================================
# 页面配置
# ============================================================

# 必须放在所有 Streamlit 页面命令之前
st.set_page_config(
    page_title="智扫通机器人智能客服",
    page_icon="🤖",
    layout="centered",
)


# 防止浏览器翻译扩展改写 Streamlit 管理的文本节点，
# 导致流式输出缺失或 removeChild 异常。
components.html(
    """
    <script>
        const doc = window.parent.document;

        doc.documentElement.setAttribute("translate", "no");
        doc.documentElement.classList.add("notranslate");

        if (window.frameElement) {
            window.frameElement.style.display = "none";
        }
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# 流式内容处理
# ============================================================

def extract_chunk_text(chunk: Any) -> str:
    """
    将 Agent 返回的流式分片转换为字符串。

    支持：
    1. 普通字符串；
    2. LangChain AIMessageChunk；
    3. content 为文本块列表的消息；
    4. (message, metadata) 格式。
    """

    if chunk is None:
        return ""

    # ReactAgent.execute_stream() 直接返回字符串
    if isinstance(chunk, str):
        return chunk

    # 某些 stream 模式可能返回：
    # (AIMessageChunk, metadata)
    if isinstance(chunk, tuple) and chunk:
        return extract_chunk_text(chunk[0])

    # 部分 LangChain 消息对象提供 text 属性或方法
    text = getattr(chunk, "text", None)

    if callable(text):
        try:
            text = text()
        except TypeError:
            text = None

    if isinstance(text, str) and text:
        return text

    # LangChain 消息通常提供 content 属性
    content = getattr(chunk, "content", None)

    if isinstance(content, str):
        return content

    # 兼容多模态消息块格式
    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                item_text = item.get("text")

                if isinstance(item_text, str):
                    text_parts.append(item_text)

        return "".join(text_parts)

    return ""


def normalize_response_stream(
    response_stream: Iterable[Any],
) -> Iterator[str]:
    """
    将 Agent 的输出统一转换为纯文本流。

    空分片不会传递给 st.write_stream。
    """

    for chunk in response_stream:
        text = extract_chunk_text(chunk)

        if text:
            yield text


# ============================================================
# 初始化页面状态
# ============================================================

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "你好，我是智扫通机器人智能客服，"
                "请问有什么可以帮助您？"
            ),
        }
    ]


# thread_id 绑定 Checkpointer 中的一条短期会话。
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())


# 测试阶段优先使用 .env 中的 DEMO_USER_ID。
# 未配置时为当前浏览器会话生成独立用户，避免不同测试者共享长期记忆。
if "user_id" not in st.session_state:
    st.session_state["user_id"] = (
        os.getenv("DEMO_USER_ID", "").strip()
        or f"test-user-{uuid.uuid4()}"
    )


# ============================================================
# 页面标题
# ============================================================

st.title("🤖 智扫通机器人智能客服")
st.divider()


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.subheader("当前会话")

    st.caption(
        f"会话 ID：{st.session_state['thread_id']}"
    )
    st.caption(
        f"用户 ID：{st.session_state['user_id']}"
    )

    if st.button(
        "清空当前会话",
        use_container_width=True,
    ):
        # 清空页面消息
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "你好，我是智扫通机器人智能客服，"
                    "请问有什么可以帮助您？"
                ),
            }
        ]

        # 生成新的 thread_id：短期消息不继承，user_id 和长期记忆保持不变。
        st.session_state["thread_id"] = str(
            uuid.uuid4()
        )

        # 清除可能残留的流式结束标记
        st.session_state.pop(
            "_stream_completed",
            None,
        )

        st.rerun()


# ============================================================
# 显示历史消息
# ============================================================

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 接收用户输入
# ============================================================

prompt = st.chat_input(
    "请输入您的问题"
)


# 流式回答结束后会执行 st.rerun()。
#
# 某些浏览器可能在重跑时保留或重放刚才的 chat_input，
# 使用该标记阻止重复调用 Agent。
#
# 放在历史消息渲染之后，可以保证重新运行时，
# 页面仍然正常显示刚刚生成的完整回答。
if st.session_state.pop(
    "_stream_completed",
    False,
):
    st.stop()


# ============================================================
# 调用 Agent
# ============================================================

if prompt:
    # 将用户消息加入页面历史
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # 当前运行中立即显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):

            # spinner 只持续到收到第一个有效文本分片
            with st.spinner("智能客服思考中..."):
                raw_stream = (
                    st.session_state["agent"]
                    .execute_stream(
                        prompt,
                        thread_id=st.session_state["thread_id"],
                        user_id=st.session_state["user_id"],
                    )
                )

                response_stream = iter(
                    normalize_response_stream(
                        raw_stream
                    )
                )

                first_chunk = next(
                    response_stream,
                    None,
                )

            # Agent 没有产生任何有效文本
            if first_chunk is None:
                raise RuntimeError(
                    "模型没有返回任何有效内容"
                )

            # 将刚才取出的第一个分片重新放回流中
            complete_response = st.write_stream(
                chain_streams(
                    (first_chunk,),
                    response_stream,
                )
            )

        # 当传入内容全部为字符串时，
        # st.write_stream 通常会返回拼接后的完整字符串。
        if not isinstance(complete_response, str):
            complete_response = "".join(
                str(item)
                for item in complete_response
            )

        complete_response = (
            complete_response.strip()
        )

        if not complete_response:
            raise RuntimeError(
                "模型返回的回答为空"
            )

        # 保存完整回答，而不是只保存最后一个 chunk
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": complete_response,
            }
        )

        # 标记当前流式回答已经完成
        st.session_state[
            "_stream_completed"
        ] = True

        # 重新运行后用完整 Markdown 渲染回答，
        # 代码块、标题、表格等格式会更加稳定
        st.rerun()

    except Exception as error:
        error_message = (
            f"处理失败：{type(error).__name__}：{error}"
        )

        # 保存错误消息，避免页面下一次重跑后消失
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": f"⚠️ {error_message}",
            }
        )

        with st.chat_message("assistant"):
            st.error(error_message)
