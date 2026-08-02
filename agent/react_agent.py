from langchain.agents import create_agent
from collections.abc import Iterator
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import rag_summarize,get_user_id,get_current_month,fetch_external_data,fill_context_for_report,get_current_date
from agent.tools.tavily_tools import web_search
from agent.tools.location_weather_agents import (
    ConversationState,
    delegate_to_location_agent,
    delegate_to_weather_agent,
)
from agent.tools.weather_tool import AgentContext
from agent.memory import SQLiteMemoryPersistence
from agent.memory.long_term import (
    delete_long_term_memory,
    list_long_term_memories,
    save_long_term_memory,
)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from dotenv import load_dotenv
load_dotenv()

class ReactAgent:
    def __init__(
        self,
        memory: SQLiteMemoryPersistence | None = None,
    ):
        self._owns_memory = memory is None
        self.memory = memory or SQLiteMemoryPersistence()
        try:
            self.agent = create_agent(
                model=chat_model,
                system_prompt=load_system_prompts(),
                tools=[
                    rag_summarize,
                    delegate_to_location_agent,
                    delegate_to_weather_agent,
                    get_user_id,
                    get_current_month,
                    get_current_date,
                    fetch_external_data,
                    fill_context_for_report,
                    save_long_term_memory,
                    list_long_term_memories,
                    delete_long_term_memory,
                    web_search,
                ],
                middleware=[monitor_tool,log_before_model,report_prompt_switch],
                state_schema=ConversationState,
                context_schema=AgentContext,
                checkpointer=self.memory.checkpointer,
                store=self.memory.store,
            )
        except BaseException:
            if self._owns_memory:
                self.memory.close()
            raise

    def execute_stream(
            self,
            query: str,
            *,
            thread_id: str,
            user_id: str,
            latitude: float | None = None,
            longitude: float | None = None,
            client_ip: str | None = None,
    ) -> Iterator[str]:
        normalized_thread_id = self._validate_runtime_id(
            "thread_id",
            thread_id,
        )
        normalized_user_id = self._validate_runtime_id(
            "user_id",
            user_id,
        )
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        input_dict = {
            "messages":[HumanMessage(query)]
        }
        config = {
            "configurable": {
                "thread_id": normalized_thread_id,
            }
        }

        #第三个参数是用来作提示词切换的，False表示正常提示词，Ture表示切换为生成报告提示词
        for chunk,metadata in self.agent.stream(
                input_dict,
                config=config,
                stream_mode="messages",
            context=AgentContext(
                report=False,
                latitude=latitude,
                longitude=longitude,
                client_ip=client_ip,
                user_id=normalized_user_id,
            )
        ):
            if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                continue

            text = chunk.text

            if text:
                yield text

    @staticmethod
    def _validate_runtime_id(name: str, value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError(f"{name} 不能为空")
        if len(normalized) > 128:
            raise ValueError(f"{name} 长度不能超过 128 个字符")
        return normalized

    def close(self) -> None:
        if self._owns_memory:
            self.memory.close()

if __name__ == "__main__":
    from uuid import uuid4

    agent = ReactAgent()
    try:
        for chunk in agent.execute_stream(
            "我目前所处的地区在哪里,准确到经纬度可以吗",
            thread_id=str(uuid4()),
            user_id="demo-user",
        ):
            print(chunk, end="", flush=True)
    finally:
        agent.close()
