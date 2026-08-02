from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.memory.long_term import (
    delete_long_term_memory,
    list_long_term_memories,
    render_user_memories_for_prompt,
    save_long_term_memory,
)
from agent.memory.persistence import SQLiteMemoryPersistence
from agent.tools.weather_tool import AgentContext


def _build_test_graph(memory: SQLiteMemoryPersistence):
    def reply(state: MessagesState):
        latest = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"reply:{latest}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(
        checkpointer=memory.checkpointer,
        store=memory.store,
    )


class SQLiteMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.short_db = str(root / "checkpoints.sqlite3")
        self.long_db = str(root / "long-term.sqlite3")
        self.memory = SQLiteMemoryPersistence(
            short_term_db=self.short_db,
            long_term_db=self.long_db,
        )

    def tearDown(self) -> None:
        self.memory.close()
        self.temp_dir.cleanup()

    def test_checkpointer_restores_only_the_same_thread(self) -> None:
        graph = _build_test_graph(self.memory)
        thread_a = {"configurable": {"thread_id": "thread-a"}}
        thread_b = {"configurable": {"thread_id": "thread-b"}}

        graph.invoke({"messages": [HumanMessage("first")]}, thread_a)
        graph.invoke({"messages": [HumanMessage("second")]}, thread_a)
        graph.invoke({"messages": [HumanMessage("other")]}, thread_b)

        messages_a = graph.get_state(thread_a).values["messages"]
        messages_b = graph.get_state(thread_b).values["messages"]
        self.assertEqual(
            [message.content for message in messages_a],
            ["first", "reply:first", "second", "reply:second"],
        )
        self.assertEqual(
            [message.content for message in messages_b],
            ["other", "reply:other"],
        )

    def test_memory_tools_do_not_expose_user_id_to_the_model(self) -> None:
        for memory_tool in (
            save_long_term_memory,
            list_long_term_memories,
            delete_long_term_memory,
        ):
            self.assertNotIn("user_id", memory_tool.args)
            self.assertNotIn("runtime", memory_tool.args)

    def test_store_shares_across_threads_but_isolates_users(self) -> None:
        user_a_runtime = SimpleNamespace(
            context=AgentContext(user_id="user-a"),
            store=self.memory.store,
        )
        same_user_new_thread_runtime = SimpleNamespace(
            context=AgentContext(user_id="user-a"),
            store=self.memory.store,
        )
        user_b_runtime = SimpleNamespace(
            context=AgentContext(user_id="user-b"),
            store=self.memory.store,
        )

        saved_a = save_long_term_memory.func(
            runtime=user_a_runtime,
            content="偏好安静模式",
            category="preference",
        )
        save_long_term_memory.func(
            runtime=user_b_runtime,
            content="设备型号为 B100",
            category="device",
        )

        user_a_memories = list_long_term_memories.func(
            runtime=same_user_new_thread_runtime,
        )
        user_b_memories = list_long_term_memories.func(
            runtime=user_b_runtime,
        )
        self.assertEqual(user_a_memories["count"], 1)
        self.assertEqual(
            user_a_memories["memories"][0]["content"],
            "偏好安静模式",
        )
        self.assertEqual(user_b_memories["count"], 1)
        self.assertEqual(
            user_b_memories["memories"][0]["content"],
            "设备型号为 B100",
        )

        cross_user_delete = delete_long_term_memory.func(
            runtime=user_b_runtime,
            memory_id=saved_a["memory_id"],
        )
        self.assertEqual(cross_user_delete["status"], "not_found")
        self.assertEqual(
            list_long_term_memories.func(runtime=user_a_runtime)["count"],
            1,
        )

    def test_sqlite_data_survives_backend_reopen(self) -> None:
        graph = _build_test_graph(self.memory)
        config = {"configurable": {"thread_id": "persistent-thread"}}
        graph.invoke({"messages": [HumanMessage("persist me")]}, config)
        runtime = SimpleNamespace(
            context=AgentContext(user_id="persistent-user"),
            store=self.memory.store,
        )
        save_long_term_memory.func(
            runtime=runtime,
            content="长期使用静音模式",
            category="preference",
        )
        self.memory.close()

        self.memory = SQLiteMemoryPersistence(
            short_term_db=self.short_db,
            long_term_db=self.long_db,
        )
        reopened_graph = _build_test_graph(self.memory)
        messages = reopened_graph.get_state(config).values["messages"]
        memories = list_long_term_memories.func(
            runtime=SimpleNamespace(
                context=AgentContext(user_id="persistent-user"),
                store=self.memory.store,
            )
        )
        self.assertEqual(messages[-1].content, "reply:persist me")
        self.assertEqual(memories["memories"][0]["content"], "长期使用静音模式")

    def test_prompt_only_contains_current_users_memories(self) -> None:
        runtime = SimpleNamespace(
            context=AgentContext(user_id="visible-user"),
            store=self.memory.store,
        )
        save_long_term_memory.func(
            runtime=runtime,
            content="拖地时偏好低水量",
            category="preference",
        )

        visible_prompt = render_user_memories_for_prompt(
            self.memory.store,
            "visible-user",
        )
        isolated_prompt = render_user_memories_for_prompt(
            self.memory.store,
            "different-user",
        )
        self.assertIn("拖地时偏好低水量", visible_prompt)
        self.assertEqual(isolated_prompt, "")


class ReactAgentMemoryWiringTests(unittest.TestCase):
    def test_react_agent_passes_thread_and_user_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SQLiteMemoryPersistence(
                short_term_db=str(Path(temp_dir) / "short.sqlite3"),
                long_term_db=str(Path(temp_dir) / "long.sqlite3"),
            )
            fake_graph = Mock()
            fake_graph.stream.return_value = [
                (AIMessageChunk(content="ok"), {}),
            ]

            try:
                with patch("agent.react_agent.create_agent", return_value=fake_graph) as factory:
                    from agent.react_agent import ReactAgent

                    agent = ReactAgent(memory=memory)
                    result = "".join(
                        agent.execute_stream(
                            "hello",
                            thread_id="thread-123",
                            user_id="user-456",
                        )
                    )

                self.assertEqual(result, "ok")
                self.assertIs(
                    factory.call_args.kwargs["checkpointer"],
                    memory.checkpointer,
                )
                self.assertIs(factory.call_args.kwargs["store"], memory.store)
                stream_kwargs = fake_graph.stream.call_args.kwargs
                self.assertEqual(
                    stream_kwargs["config"]["configurable"]["thread_id"],
                    "thread-123",
                )
                self.assertEqual(stream_kwargs["context"].user_id, "user-456")
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
