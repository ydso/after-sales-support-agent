"""Chroma 向量库服务及按源文件管理的增量导入。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embedding_model
from rag.document_manifest import DocumentManifest, ManifestEntry
from utils.config_handler import chroma_config
from utils.file_handler import (
    get_file_sha256_hex,
    listdir_with_allowed_type,
    pdf_loader,
    text_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


_HNSW_SPACES = {"l2", "cosine", "ip"}


def _resolve_persist_directory() -> Path:
    """解析 Chroma 目录，并阻止 Windows 原生 HNSW 使用非 ASCII 路径。"""

    configured_path = str(chroma_config["persist_directory"]).strip()
    environment_name = str(
        chroma_config.get("persist_directory_env", "CHROMA_PERSIST_DIRECTORY")
    ).strip()
    raw_path = os.getenv(environment_name, configured_path).strip()
    if not raw_path:
        raise ValueError("Chroma 持久化目录不能为空")

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(get_abs_path(raw_path))
    path = path.resolve()

    if os.name == "nt" and not str(path).isascii():
        raise ValueError(
            "Windows 下 Chroma HNSW 持久化目录必须是纯英文/ASCII 路径，"
            f"当前路径为: {path}。请修改 config/chroma.yml，或设置 "
            f"{environment_name}。"
        )
    return path


def _load_hnsw_configuration() -> dict[str, Any]:
    """读取并校验传给 Chroma 新集合的 HNSW 配置。"""

    raw_config = chroma_config.get("hnsw")
    if not isinstance(raw_config, Mapping):
        raise ValueError("config/chroma.yml 中必须配置 hnsw")

    space = str(raw_config.get("space", "l2")).strip().lower()
    if space not in _HNSW_SPACES:
        raise ValueError(
            f"hnsw.space 必须是 {sorted(_HNSW_SPACES)} 之一，当前为: {space}"
        )

    values: dict[str, int] = {}
    for key in ("batch_size", "sync_threshold"):
        value = raw_config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"hnsw.{key} 必须是大于等于 1 的整数")
        values[key] = value
    if values["batch_size"] > values["sync_threshold"]:
        raise ValueError("hnsw.batch_size 不能大于 hnsw.sync_threshold")

    return {
        "space": space,
        "batch_size": values["batch_size"],
        "sync_threshold": values["sync_threshold"],
    }


@dataclass
class IngestionSummary:
    """一次知识库同步的可观测结果。"""

    imported: int = 0
    updated: int = 0
    unchanged: int = 0
    repaired: int = 0
    deleted: int = 0
    legacy_vectors_deleted: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class VectorStoreService:
    """维护 Chroma 检索器，并将知识文件同步到向量库。"""

    def __init__(self) -> None:
        self.data_root = Path(get_abs_path(chroma_config["data_path"])).resolve()
        self.allowed_extensions = {
            f".{file_type.lower().lstrip('.')}"
            for file_type in chroma_config["allow_knowledge_file_type"]
        }
        self.manifest_path = Path(
            get_abs_path(chroma_config["document_manifest_db"])
        ).resolve()
        self.persist_directory = _resolve_persist_directory()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.hnsw_configuration = _load_hnsw_configuration()
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=str(self.persist_directory),
            collection_configuration={"hnsw": self.hnsw_configuration},
        )
        self._verify_active_collection_configuration()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )
        logger.info(
            "[初始化向量库] persist_directory=%s hnsw=%s",
            self.persist_directory,
            self.hnsw_configuration,
        )

    def _verify_active_collection_configuration(self) -> None:
        """避免已有集合静默忽略仅在创建时传入的关键 HNSW 参数。"""

        collection_configuration = self.vector_store._collection.configuration
        actual_hnsw = collection_configuration.get("hnsw") or {}
        expected_space = self.hnsw_configuration["space"]
        expected_sync_threshold = self.hnsw_configuration["sync_threshold"]
        if (
            actual_hnsw.get("space") != expected_space
            or actual_hnsw.get("sync_threshold") != expected_sync_threshold
        ):
            raise RuntimeError(
                "现有 Chroma 集合的 HNSW 配置与 config/chroma.yml 不一致。"
                "创建参数不会覆盖旧集合，请停止应用、备份并重建该向量库。"
                f" expected={self.hnsw_configuration} actual={actual_hnsw}"
            )

    def get_retriever(self):
        """返回按配置中的 ``k`` 检索 Chroma 的 Retriever。"""

        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_config["k"]}
        )

    def _source_key(self, file_path: Path) -> str:
        """使用相对 data 目录的 POSIX 路径作为跨运行稳定的文件身份。"""

        return file_path.resolve().relative_to(self.data_root).as_posix()

    @staticmethod
    def _vector_id(
        source_key: str,
        content_sha256: str,
        chunk_index: int,
    ) -> str:
        """内容版本和分块序号共同生成确定性 ID，保证失败重试幂等。"""

        identity = f"{source_key}\0{content_sha256}\0{chunk_index}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_ids(result: Mapping[str, Any]) -> set[str]:
        """兼容 Chroma 返回的一维或嵌套 ID 列表。"""

        raw_ids = result.get("ids") or []
        ids: set[str] = set()
        for item in raw_ids:
            if isinstance(item, str):
                ids.add(item)
            elif isinstance(item, Sequence):
                ids.update(value for value in item if isinstance(value, str))
        return ids

    def _get_vector_ids(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> set[str]:
        if ids is not None and not ids:
            return set()

        kwargs: dict[str, Any] = {"include": []}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where is not None:
            kwargs["where"] = dict(where)
        return self._extract_ids(self.vector_store.get(**kwargs))

    def _existing_source_ids(self, file_path: Path, source_key: str) -> set[str]:
        """同时发现新清单分块和旧 MD5 导入留下的绝对路径分块。"""

        managed_ids = self._get_vector_ids(where={"ingestion_source": source_key})
        legacy_ids = self._get_vector_ids(where={"source": str(file_path.resolve())})
        return managed_ids | legacy_ids

    def _delete_vector_ids(self, vector_ids: Sequence[str] | set[str]) -> None:
        ids = sorted(set(vector_ids))
        if ids:
            self.vector_store.delete(ids=ids)

    @staticmethod
    def _load_file_documents(file_path: Path) -> list[Document]:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return text_loader(file_path)
        if suffix == ".pdf":
            return pdf_loader(file_path)
        raise ValueError(f"不支持的知识库文件类型: {file_path}")

    def _prepare_chunks(
        self,
        *,
        file_path: Path,
        source_key: str,
        content_sha256: str,
    ) -> tuple[list[Document], list[str]]:
        documents = self._load_file_documents(file_path)

        # 文件可能在解析期间被外部程序改写。摘要不一致时本轮放弃，避免
        # “清单记录旧摘要、向量内容却来自新文件”的错配状态。
        if get_file_sha256_hex(file_path) != content_sha256:
            raise RuntimeError(f"文件在导入过程中发生变化，请重试: {file_path}")

        chunks = self.splitter.split_documents(documents)
        vector_ids: list[str] = []
        for chunk_index, chunk in enumerate(chunks):
            vector_id = self._vector_id(
                source_key,
                content_sha256,
                chunk_index,
            )
            chunk.metadata = {
                **chunk.metadata,
                "ingestion_source": source_key,
                "content_sha256": content_sha256,
                "chunk_index": chunk_index,
            }
            vector_ids.append(vector_id)
        return chunks, vector_ids

    def _remove_deleted_files(
        self,
        *,
        manifest: DocumentManifest,
        current_source_keys: set[str],
        summary: IngestionSummary,
    ) -> None:
        """删除已从 data 目录移走的文件所对应的全部旧向量。"""

        for entry in manifest.list_all():
            if entry.source_path in current_source_keys:
                continue
            try:
                self._delete_vector_ids(entry.vector_ids)
                manifest.delete(entry.source_path)
                summary.deleted += 1
                logger.info(
                    "[同步知识库] 源文件已删除，旧向量已清理: %s",
                    entry.source_path,
                )
            except Exception:
                summary.failed += 1
                logger.exception(
                    "[同步知识库] 清理已删除文件失败，保留清单以便下次重试: %s",
                    entry.source_path,
                )

    def _sync_file(
        self,
        *,
        file_path: Path,
        source_key: str,
        manifest: DocumentManifest,
        summary: IngestionSummary,
    ) -> None:
        content_sha256 = get_file_sha256_hex(file_path)
        entry = manifest.get(source_key)
        existing_source_ids = self._existing_source_ids(file_path, source_key)

        if entry is not None and entry.content_sha256 == content_sha256:
            expected_ids = set(entry.vector_ids)
            present_ids = self._get_vector_ids(ids=entry.vector_ids)
            if present_ids == expected_ids:
                # 即使文件版本未变，也清理由旧 MD5 流程或中断写入遗留的额外分块。
                self._delete_vector_ids(existing_source_ids - expected_ids)
                stat = file_path.stat()
                manifest.replace(
                    source_path=source_key,
                    content_sha256=content_sha256,
                    vector_ids=entry.vector_ids,
                    file_size=stat.st_size,
                    modified_ns=stat.st_mtime_ns,
                )
                summary.unchanged += 1
                logger.info("[同步知识库] 内容未变化，跳过: %s", source_key)
                return

        chunks, vector_ids = self._prepare_chunks(
            file_path=file_path,
            source_key=source_key,
            content_sha256=content_sha256,
        )
        if chunks:
            # 先 upsert 新版本，再删除旧版本。若流程中断，确定性 ID 允许下次安全重试。
            self.vector_store.add_documents(documents=chunks, ids=vector_ids)

        self._delete_vector_ids(existing_source_ids - set(vector_ids))
        stat = file_path.stat()
        manifest.replace(
            source_path=source_key,
            content_sha256=content_sha256,
            vector_ids=vector_ids,
            file_size=stat.st_size,
            modified_ns=stat.st_mtime_ns,
        )

        if entry is None:
            summary.imported += 1
            action = "首次导入"
        elif entry.content_sha256 != content_sha256:
            summary.updated += 1
            action = "内容更新"
        else:
            summary.repaired += 1
            action = "缺失分块修复"
        logger.info(
            "[同步知识库] %s成功: %s，共 %d 个分块",
            action,
            source_key,
            len(vector_ids),
        )

    def _is_managed_legacy_source(
        self,
        source: object,
        current_absolute_paths: set[Path],
    ) -> bool:
        """判断旧 ``source`` 元数据是否属于 data 目录中的已删除知识文件。"""

        if not isinstance(source, str) or not source:
            return False
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        try:
            resolved = source_path.resolve()
            resolved.relative_to(self.data_root)
        except (OSError, ValueError):
            return False
        return (
            resolved.suffix.lower() in self.allowed_extensions
            and resolved not in current_absolute_paths
        )

    def _cleanup_legacy_orphans(
        self,
        *,
        current_source_keys: set[str],
        current_absolute_paths: set[Path],
        summary: IngestionSummary,
    ) -> None:
        """首次迁移时清理旧 MD5 机制无法关联到清单的孤儿向量。"""

        result = self.vector_store.get(include=["metadatas"])
        ids = list(result.get("ids") or [])
        metadatas = list(result.get("metadatas") or [])
        stale_ids: set[str] = set()
        for vector_id, metadata in zip(ids, metadatas, strict=False):
            if not isinstance(vector_id, str) or not isinstance(metadata, Mapping):
                continue
            ingestion_source = metadata.get("ingestion_source")
            if isinstance(ingestion_source, str):
                if ingestion_source not in current_source_keys:
                    stale_ids.add(vector_id)
                continue
            if self._is_managed_legacy_source(
                metadata.get("source"),
                current_absolute_paths,
            ):
                stale_ids.add(vector_id)

        self._delete_vector_ids(stale_ids)
        summary.legacy_vectors_deleted += len(stale_ids)
        if stale_ids:
            logger.info(
                "[同步知识库] 已清理 %d 个旧导入机制遗留的孤儿向量",
                len(stale_ids),
            )

    def load_documents(self) -> IngestionSummary:
        """
        将 data 目录与 Chroma 做文件级增量同步。

        - 新文件：导入并写入清单；
        - 内容变化：写入新分块后删除该文件旧分块；
        - 内容未变：跳过嵌入，仅校验分块完整性；
        - 文件删除：删除清单记录和对应的全部旧分块。
        """

        summary = IngestionSummary()
        allowed_files = tuple(
            Path(path).resolve()
            for path in listdir_with_allowed_type(
                self.data_root,
                chroma_config["allow_knowledge_file_type"],
            )
        )
        current_source_keys = {self._source_key(path) for path in allowed_files}
        current_absolute_paths = set(allowed_files)

        with DocumentManifest(self.manifest_path) as manifest:
            with manifest.sync_transaction():
                self._remove_deleted_files(
                    manifest=manifest,
                    current_source_keys=current_source_keys,
                    summary=summary,
                )
                for file_path in allowed_files:
                    source_key = self._source_key(file_path)
                    try:
                        self._sync_file(
                            file_path=file_path,
                            source_key=source_key,
                            manifest=manifest,
                            summary=summary,
                        )
                    except Exception:
                        summary.failed += 1
                        logger.exception(
                            "[同步知识库] 文件同步失败，将在下次运行重试: %s",
                            source_key,
                        )

                try:
                    self._cleanup_legacy_orphans(
                        current_source_keys=current_source_keys,
                        current_absolute_paths=current_absolute_paths,
                        summary=summary,
                    )
                except Exception:
                    summary.failed += 1
                    logger.exception(
                        "[同步知识库] 遗留孤儿向量清理失败，将在下次运行重试"
                    )

        logger.info("[同步知识库] 完成: %s", summary.as_dict())
        return summary


if __name__ == "__main__":
    # 初始化并加载知识库同步服务
    service = VectorStoreService()
    # 同步知识库
    print(service.load_documents().as_dict())
