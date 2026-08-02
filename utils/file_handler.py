"""知识库文件发现、摘要计算和内容加载。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


# 每次读取文件的大小：1 MB。
# 计算文件哈希时不会一次性将整个文件读取到内存，
# 可以避免大文件占用过多内存。
_HASH_READ_SIZE = 1024 * 1024


def get_file_sha256_hex(file_path: str | Path) -> str:
    """流式计算文件 SHA-256；路径无效或读取失败时直接抛出异常。"""

    # 统一转换为 Path 对象，方便后续进行路径检查和文件操作。
    path = Path(file_path)

    # 先检查路径是否存在，避免在打开文件时才出现不明确的错误。
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在: {path}")

    # 路径可能是目录，因此还需要确认它确实是文件。
    if not path.is_file():
        raise ValueError(f"知识库路径不是文件: {path}")

    # 创建 SHA-256 摘要计算器。
    digest = hashlib.sha256()

    # 每次读取 1 MB 数据，逐块更新摘要。
    # 海象运算符 := 会先读取数据，再判断数据是否为空。
    with path.open("rb") as file:
        while chunk := file.read(_HASH_READ_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def listdir_with_allowed_type(
    path: str | Path,
    allowed_types: Iterable[str],
) -> tuple[str, ...]:
    """按稳定顺序返回目录第一层中允许导入的文件绝对路径。"""

    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"知识库数据目录不存在或不是目录: {directory}")

    allowed_suffixes = {
        f".{file_type.lower().lstrip('.')}" for file_type in allowed_types
    }
    files = sorted(
        (
            item.resolve()
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in allowed_suffixes
        ),
        key=lambda item: item.name.casefold(),
    )
    return tuple(str(item) for item in files)


def pdf_loader(file_path: str | Path, password: str | None = None) -> list[Document]:
    """加载 PDF；加密 PDF 可传入密码。"""

    return PyPDFLoader(str(file_path), password=password).load()


def text_loader(file_path: str | Path) -> list[Document]:
    """以 UTF-8 加载纯文本文件。"""

    return TextLoader(str(file_path), encoding="utf-8").load()
