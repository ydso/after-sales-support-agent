"""SQLite 文档导入清单：维护源文件、内容摘要与 Chroma 分块 ID 的映射。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class ManifestEntry:
    """一个源文件最近一次成功同步到 Chroma 的完整状态。"""

    source_path: str
    content_sha256: str
    vector_ids: tuple[str, ...]
    file_size: int
    modified_ns: int
    updated_at: str


class DocumentManifest:
    """
    管理文档导入状态。

    清单与 Chroma 无法共享数据库事务，因此调用方必须遵守：先完成
    Chroma upsert/delete，再更新清单。确定性向量 ID 使中断后的重试保持幂等。
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._setup()

    def _setup(self) -> None:
        """创建文件表和分块表；分块随文件记录删除。"""

        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_files (
                source_path TEXT PRIMARY KEY,
                content_sha256 TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                source_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                vector_id TEXT NOT NULL UNIQUE,
                PRIMARY KEY (source_path, chunk_index),
                FOREIGN KEY (source_path)
                    REFERENCES document_files(source_path)
                    ON DELETE CASCADE
            );
            """
        )
        self.connection.commit()

    @contextmanager
    def sync_transaction(self) -> Iterator[None]:
        """串行化清单写入，并在同步异常时回滚清单状态。"""

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def get(self, source_path: str) -> ManifestEntry | None:
        row = self.connection.execute(
            """
            SELECT source_path, content_sha256, file_size, modified_ns, updated_at
            FROM document_files
            WHERE source_path = ?
            """,
            (source_path,),
        ).fetchone()
        if row is None:
            return None

        vector_ids = tuple(
            chunk_row[0]
            for chunk_row in self.connection.execute(
                """
                SELECT vector_id
                FROM document_chunks
                WHERE source_path = ?
                ORDER BY chunk_index
                """,
                (source_path,),
            ).fetchall()
        )
        return ManifestEntry(
            source_path=row[0],
            content_sha256=row[1],
            vector_ids=vector_ids,
            file_size=row[2],
            modified_ns=row[3],
            updated_at=row[4],
        )

    def list_all(self) -> list[ManifestEntry]:
        source_paths = [
            row[0]
            for row in self.connection.execute(
                "SELECT source_path FROM document_files ORDER BY source_path"
            ).fetchall()
        ]
        return [
            entry
            for source_path in source_paths
            if (entry := self.get(source_path)) is not None
        ]

    def replace(
        self,
        *,
        source_path: str,
        content_sha256: str,
        vector_ids: Sequence[str],
        file_size: int,
        modified_ns: int,
    ) -> None:
        """原子替换单个源文件的摘要和完整分块 ID 列表。"""

        updated_at = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT INTO document_files (
                source_path,
                content_sha256,
                file_size,
                modified_ns,
                updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                content_sha256 = excluded.content_sha256,
                file_size = excluded.file_size,
                modified_ns = excluded.modified_ns,
                updated_at = excluded.updated_at
            """,
            (
                source_path,
                content_sha256,
                file_size,
                modified_ns,
                updated_at,
            ),
        )
        self.connection.execute(
            "DELETE FROM document_chunks WHERE source_path = ?",
            (source_path,),
        )
        self.connection.executemany(
            """
            INSERT INTO document_chunks (source_path, chunk_index, vector_id)
            VALUES (?, ?, ?)
            """,
            (
                (source_path, chunk_index, vector_id)
                for chunk_index, vector_id in enumerate(vector_ids)
            ),
        )

    def delete(self, source_path: str) -> None:
        """删除文件记录；外键级联删除其所有分块映射。"""

        self.connection.execute(
            "DELETE FROM document_files WHERE source_path = ?",
            (source_path,),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> DocumentManifest:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
