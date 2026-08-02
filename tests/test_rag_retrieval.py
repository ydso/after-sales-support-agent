from __future__ import annotations

import unittest
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag.retrieval import HybridRetriever, QueryRewriter, RetrievalConfig


def make_document(content: str, chunk_index: int) -> Document:
    """创建带稳定导入元数据的测试文档。"""

    return Document(
        page_content=content,
        metadata={
            "ingestion_source": "test.txt",
            "content_sha256": "hash",
            "chunk_index": chunk_index,
        },
    )


class FakeRewriteModel:
    """返回固定查询改写结果。"""

    def __init__(self, content: str | Exception) -> None:
        self.content = content

    def invoke(self, _messages: Any) -> AIMessage:
        if isinstance(self.content, Exception):
            raise self.content
        return AIMessage(content=self.content)


class FakeVectorStore:
    """为每个查询返回预设向量结果并暴露关键词语料。"""

    def __init__(
        self,
        routes: dict[str, list[Document]],
        corpus: list[Document],
    ) -> None:
        self.routes = routes
        self.corpus = corpus
        self.queries: list[str] = []

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.queries.append(query)
        return self.routes.get(query, [])[:k]

    def get(self, include: list[str]) -> dict[str, Any]:
        del include
        return {
            "ids": [f"id-{index}" for index in range(len(self.corpus))],
            "documents": [document.page_content for document in self.corpus],
            "metadatas": [document.metadata for document in self.corpus],
        }


def retrieval_config(final_k: int = 2) -> RetrievalConfig:
    """构造适合单元测试的混合检索配置。"""

    return RetrievalConfig(
        final_k=final_k,
        max_rewrites=2,
        dense_candidates_per_query=4,
        keyword_candidates=4,
        fusion_candidates=8,
        rrf_k=60,
        original_query_weight=1.0,
        rewritten_query_weight=0.85,
        keyword_weight=0.8,
        fusion_score_weight=0.8,
        semantic_score_weight=0.15,
        keyword_score_weight=0.05,
    )


class QueryRewriterTests(unittest.TestCase):
    def test_original_query_is_kept_and_valid_rewrites_are_deduplicated(self) -> None:
        model = FakeRewriteModel(
            '{"queries":["扫地机器人 E12 无法充电的排查方法",'
            '"扫地机器人 E12 无法充电的排查方法",'
            '"充电故障"]}'
        )
        rewriter = QueryRewriter(model=model, max_rewrites=2)

        queries = rewriter.rewrite("扫地机器人 E12 无法充电")

        self.assertEqual(
            queries,
            [
                "扫地机器人 E12 无法充电",
                "扫地机器人 E12 无法充电的排查方法",
            ],
        )

    def test_model_failure_falls_back_to_original_query(self) -> None:
        rewriter = QueryRewriter(
            model=FakeRewriteModel(RuntimeError("offline")),
            max_rewrites=2,
        )

        self.assertEqual(rewriter.rewrite("机器人无法回充"), ["机器人无法回充"])


class HybridRetrieverTests(unittest.TestCase):
    def test_multiple_routes_are_fused_and_relevant_document_is_ranked_first(self) -> None:
        generic = make_document("扫地机器人日常维护和清洁方法。", 0)
        relevant = make_document("E12 表示机器人无法充电，应检查充电触点。", 1)
        secondary = make_document("充电座应放在平整且无遮挡的位置。", 2)
        original = "扫地机器人 E12 无法充电"
        rewritten = "扫地机器人 E12 无法充电时检查充电触点"
        vector_store = FakeVectorStore(
            routes={
                original: [generic, relevant],
                rewritten: [relevant, secondary],
            },
            corpus=[generic, relevant, secondary],
        )
        rewriter = QueryRewriter(
            model=FakeRewriteModel(f'{{"queries":["{rewritten}"]}}'),
            max_rewrites=2,
        )
        retriever = HybridRetriever(
            vector_store=vector_store,
            query_rewriter=rewriter,
            config=retrieval_config(),
        )

        documents = retriever.invoke(original)

        self.assertEqual(vector_store.queries, [original, rewritten])
        self.assertEqual(documents[0].page_content, relevant.page_content)
        self.assertEqual(len(documents), 2)

    def test_original_semantic_distance_prevents_rewrite_drift(self) -> None:
        relevant = make_document("拖地后留下水痕时应调小出水量并清洗拖布。", 0)
        generic = make_document("可以在 APP 中设置不同房间的扫拖参数。", 1)
        original = "扫拖机器人留下水痕怎么处理"
        rewritten = "扫拖机器人 APP 参数设置"

        class ScoredVectorStore(FakeVectorStore):
            def similarity_search_with_score(
                self,
                query: str,
                k: int,
            ) -> list[tuple[Document, float]]:
                self.queries.append(query)
                return [(relevant, 0.1), (generic, 0.5)][:k]

        vector_store = ScoredVectorStore(
            routes={rewritten: [generic, relevant]},
            corpus=[generic, relevant],
        )
        retriever = HybridRetriever(
            vector_store=vector_store,
            query_rewriter=QueryRewriter(
                model=FakeRewriteModel(f'{{"queries":["{rewritten}"]}}'),
                max_rewrites=2,
            ),
            config=retrieval_config(),
        )

        documents = retriever.invoke(original)

        self.assertEqual(vector_store.queries, [original, rewritten])
        self.assertEqual(documents[0], relevant)

    def test_keyword_failure_does_not_remove_dense_results(self) -> None:
        relevant = make_document("机器人无法回充时应清洁充电触点。", 0)

        class BrokenKeywordStore(FakeVectorStore):
            def get(self, include: list[str]) -> dict[str, Any]:
                del include
                raise RuntimeError("keyword index unavailable")

        vector_store = BrokenKeywordStore(
            routes={"机器人无法回充": [relevant]},
            corpus=[],
        )
        retriever = HybridRetriever(
            vector_store=vector_store,
            query_rewriter=QueryRewriter(
                model=FakeRewriteModel('{"queries":[]}'),
                max_rewrites=2,
            ),
            config=retrieval_config(final_k=1),
        )

        documents = retriever.invoke("机器人无法回充")

        self.assertEqual(documents, [relevant])


if __name__ == "__main__":
    unittest.main()
