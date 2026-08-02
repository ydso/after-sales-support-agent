"""SQLite checkpointer 与 store 的连接和生命周期管理。"""

from __future__ import annotations
from pathlib import Path
from types import TracebackType
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from utils.config_handler import agent_config
from utils.path_tool import get_project_root


def _resolve_database_path(value: str) -> str:
    if value == ":memory:":
        return value

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(get_project_root()) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


class SQLiteMemoryPersistence:
    """
    分别管理线程级 checkpoint 和用户级长期记忆数据库。

    一个实例应在 ReactAgent 的整个生命周期内保持存活，以保证底层
    SQLite 连接不会在 Agent 运行期间被提前关闭。
    """

    def __init__(
        self,
        *,
        short_term_db: str | None = None,
        long_term_db: str | None = None,
    ) -> None:
        self.short_term_db = _resolve_database_path(
            short_term_db
            or agent_config.get(
                "short_term_memory_db",
                "data/memory/checkpoints.sqlite3",
            )
        )
        self.long_term_db = _resolve_database_path(
            long_term_db
            or agent_config.get(
                "long_term_memory_db",
                "data/memory/long_term.sqlite3",
            )
        )

        self._checkpointer_context = SqliteSaver.from_conn_string(
            self.short_term_db
        )
        self.checkpointer = self._checkpointer_context.__enter__()

        self._store_context = None
        try:
            self._store_context = SqliteStore.from_conn_string(
                self.long_term_db
            )
            self.store = self._store_context.__enter__()
            self.checkpointer.setup()
            self.store.setup()
        except BaseException:
            if self._store_context is not None:
                self._store_context.__exit__(None, None, None)
            self._checkpointer_context.__exit__(None, None, None)
            raise

        self._closed = False

    def delete_thread(self, thread_id: str) -> None:
        normalized = thread_id.strip()
        if not normalized:
            raise ValueError("thread_id 不能为空")
        self.checkpointer.delete_thread(normalized)

    def close(self) -> None:
        if self._closed:
            return
        self._store_context.__exit__(None, None, None)
        self._checkpointer_context.__exit__(None, None, None)
        self._closed = True

    def __enter__(self) -> SQLiteMemoryPersistence:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
