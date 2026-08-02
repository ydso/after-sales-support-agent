from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import vector_store as vector_store_module
from rag.document_manifest import DocumentManifest
from rag.vector_store import VectorStoreService


class FakeVectorStore:
    """只实现增量同步所需的 Chroma 接口。"""

    def __init__(self) -> None:
        self.records: dict[str, Document] = {}

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str],
        **_: Any,
    ) -> list[str]:
        for vector_id, document in zip(ids, documents, strict=True):
            self.records[vector_id] = document
        return list(ids)

    def delete(self, ids: Sequence[str] | None = None, **_: Any) -> None:
        for vector_id in ids or ():
            self.records.pop(vector_id, None)

    def get(
        self,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        selected: list[tuple[str, Document]] = []
        requested_ids = set(ids) if ids is not None else None
        for vector_id, document in self.records.items():
            if requested_ids is not None and vector_id not in requested_ids:
                continue
            if where is not None and any(
                document.metadata.get(key) != value
                for key, value in where.items()
            ):
                continue
            selected.append((vector_id, document))

        return {
            "ids": [vector_id for vector_id, _ in selected],
            "metadatas": [document.metadata for _, document in selected]
            if include and "metadatas" in include
            else None,
        }


def make_service(data_root: Path) -> VectorStoreService:
    service = VectorStoreService.__new__(VectorStoreService)
    service.data_root = data_root.resolve()
    service.allowed_extensions = {".txt", ".pdf"}
    service.manifest_path = data_root / "index" / "document_manifest.sqlite3"
    service.vector_store = FakeVectorStore()
    service.splitter = RecursiveCharacterTextSplitter(
        chunk_size=12,
        chunk_overlap=2,
        separators=["\n", " ", ""],
        length_function=len,
    )
    return service


class ChromaPersistenceConfigurationTests(unittest.TestCase):
    def test_service_passes_configured_hnsw_values_to_new_collection(self) -> None:
        fake_store = Mock()
        fake_store._collection = SimpleNamespace(
            configuration={
                "hnsw": {
                    "space": "l2",
                    "sync_threshold": 10,
                }
            }
        )

        with tempfile.TemporaryDirectory(dir=r"C:\Temp") as directory:
            with (
                patch.object(vector_store_module, "Chroma", return_value=fake_store) as chroma,
                patch.dict(
                    vector_store_module.chroma_config,
                    {
                        "persist_directory": directory,
                        "persist_directory_env": "TEST_CHROMA_PERSIST_DIRECTORY",
                        "hnsw": {
                            "space": "l2",
                            "batch_size": 10,
                            "sync_threshold": 10,
                        },
                    },
                    clear=False,
                ),
                patch.dict(
                    os.environ,
                    {"TEST_CHROMA_PERSIST_DIRECTORY": directory},
                    clear=False,
                ),
            ):
                service = VectorStoreService()

        self.assertEqual(service.persist_directory, Path(directory).resolve())
        chroma.assert_called_once()
        self.assertEqual(
            chroma.call_args.kwargs["collection_configuration"],
            {
                "hnsw": {
                    "space": "l2",
                    "batch_size": 10,
                    "sync_threshold": 10,
                }
            },
        )

    @unittest.skipUnless(
        os.name == "nt" and Path(r"C:\Temp").is_dir(),
        "该回归测试验证 Windows HNSW 的 ASCII 持久化路径",
    )
    def test_ascii_path_hnsw_index_survives_process_restart(self) -> None:
        writer = r'''
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

store = Chroma(
    collection_name="persist_contract",
    embedding_function=FakeEmbeddings(size=8),
    persist_directory=os.environ["CHROMA_CONTRACT_PATH"],
    collection_configuration={
        "hnsw": {"space": "l2", "batch_size": 10, "sync_threshold": 10}
    },
)
store.add_documents(
    [Document(page_content=f"document {index}") for index in range(20)],
    ids=[f"id-{index}" for index in range(20)],
)
print(store._collection.count())
'''
        reader = r'''
import os
from langchain_chroma import Chroma
from langchain_core.embeddings import FakeEmbeddings

store = Chroma(
    collection_name="persist_contract",
    embedding_function=FakeEmbeddings(size=8),
    persist_directory=os.environ["CHROMA_CONTRACT_PATH"],
)
print(store._collection.count())
print(len(store.similarity_search("test", k=1)))
'''

        with tempfile.TemporaryDirectory(dir=r"C:\Temp") as directory:
            environment = {
                **os.environ,
                "CHROMA_CONTRACT_PATH": directory,
                "PYTHONUTF8": "1",
            }
            writer_result = subprocess.run(
                [sys.executable, "-c", writer],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(writer_result.stdout.strip(), "20")

            hnsw_files = {
                path.name
                for path in Path(directory).rglob("*")
                if path.is_file() and path.name != "chroma.sqlite3"
            }
            self.assertTrue(
                {
                    "data_level0.bin",
                    "header.bin",
                    "index_metadata.pickle",
                    "length.bin",
                    "link_lists.bin",
                }.issubset(hnsw_files)
            )

            reader_result = subprocess.run(
                [sys.executable, "-c", reader],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(reader_result.stdout.strip().splitlines(), ["20", "1"])

    def test_non_ascii_windows_persist_path_is_rejected_early(self) -> None:
        with (
            patch.dict(
                vector_store_module.chroma_config,
                {
                    "persist_directory": r"D:\知识库\chroma_db",
                    "persist_directory_env": "TEST_CHROMA_PERSIST_DIRECTORY",
                },
                clear=False,
            ),
            patch.dict(
                os.environ,
                {"TEST_CHROMA_PERSIST_DIRECTORY": r"D:\知识库\chroma_db"},
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "ASCII"):
                vector_store_module._resolve_persist_directory()


class DocumentManifestTests(unittest.TestCase):
    def test_replace_and_delete_keep_file_chunk_mapping_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.sqlite3"
            with DocumentManifest(manifest_path) as manifest:
                with manifest.sync_transaction():
                    manifest.replace(
                        source_path="guide.txt",
                        content_sha256="hash-1",
                        vector_ids=["chunk-1", "chunk-2"],
                        file_size=20,
                        modified_ns=100,
                    )
                entry = manifest.get("guide.txt")
                self.assertIsNotNone(entry)
                self.assertEqual(entry.vector_ids, ("chunk-1", "chunk-2"))

                with manifest.sync_transaction():
                    manifest.delete("guide.txt")
                self.assertIsNone(manifest.get("guide.txt"))


class IncrementalIngestionTests(unittest.TestCase):
    def test_new_unchanged_modified_and_deleted_file_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            source_file = data_root / "guide.txt"
            source_file.write_text(
                "旧版本知识内容，需要被分成多个向量块。",
                encoding="utf-8",
            )
            service = make_service(data_root)

            first = service.load_documents()
            first_ids = set(service.vector_store.records)
            self.assertEqual(first.imported, 1)
            self.assertTrue(first_ids)

            second = service.load_documents()
            self.assertEqual(second.unchanged, 1)
            self.assertEqual(set(service.vector_store.records), first_ids)

            source_file.write_text(
                "新版本知识内容，旧版本对应的向量必须全部删除。",
                encoding="utf-8",
            )
            third = service.load_documents()
            third_ids = set(service.vector_store.records)
            self.assertEqual(third.updated, 1)
            self.assertTrue(third_ids)
            self.assertTrue(first_ids.isdisjoint(third_ids))

            source_file.unlink()
            fourth = service.load_documents()
            self.assertEqual(fourth.deleted, 1)
            self.assertEqual(service.vector_store.records, {})
            with DocumentManifest(service.manifest_path) as manifest:
                self.assertEqual(manifest.list_all(), [])

    def test_missing_chunk_is_repaired_even_when_file_hash_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            source_file = data_root / "guide.txt"
            source_file.write_text("需要生成多个分块的测试内容。" * 4, encoding="utf-8")
            service = make_service(data_root)

            service.load_documents()
            missing_id = next(iter(service.vector_store.records))
            service.vector_store.records.pop(missing_id)

            repaired = service.load_documents()

            self.assertEqual(repaired.repaired, 1)
            with DocumentManifest(service.manifest_path) as manifest:
                entry = manifest.get("guide.txt")
                self.assertIsNotNone(entry)
                self.assertEqual(
                    set(entry.vector_ids),
                    set(service.vector_store.records),
                )

    def test_first_migration_removes_legacy_and_deleted_file_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            current_file = data_root / "current.txt"
            deleted_file = data_root / "deleted.txt"
            outside_file = data_root.parent / "outside.txt"
            current_file.write_text("当前仍存在的知识文件。", encoding="utf-8")
            service = make_service(data_root)

            service.vector_store.records.update(
                {
                    "legacy-current": Document(
                        page_content="旧版本",
                        metadata={"source": str(current_file.resolve())},
                    ),
                    "legacy-deleted": Document(
                        page_content="已删除文件",
                        metadata={"source": str(deleted_file.resolve())},
                    ),
                    "unmanaged": Document(
                        page_content="不属于 data 目录",
                        metadata={"source": str(outside_file.resolve())},
                    ),
                }
            )

            result = service.load_documents()

            self.assertEqual(result.imported, 1)
            self.assertNotIn("legacy-current", service.vector_store.records)
            self.assertNotIn("legacy-deleted", service.vector_store.records)
            self.assertIn("unmanaged", service.vector_store.records)
            self.assertEqual(result.legacy_vectors_deleted, 1)


if __name__ == "__main__":
    unittest.main()
