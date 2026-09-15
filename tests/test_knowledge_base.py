from __future__ import annotations

import unittest

from prd_agent.contracts import GenerationRequest, SourceType
from prd_agent.knowledge_base import (
    AgentCoreKnowledgeBaseRetriever,
    GatewayPassage,
    KnowledgeBaseRetrievalError,
)


class FakeGateway:
    def __init__(self, results_by_query) -> None:
        self.results_by_query = results_by_query
        self.calls = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.results_by_query.get(kwargs["query"], [])


class FakeManifestProvider:
    def __init__(self) -> None:
        self.calls = []

    async def load(self, **kwargs):
        self.calls.append(kwargs)
        return {"src/imports.py": ["ImportService", "ImportService.create"]}


class KnowledgeBaseRetrieverTest(unittest.IsolatedAsyncioTestCase):
    def request(self) -> GenerationRequest:
        return GenerationRequest(
            project_id="project-1",
            requirement_text="Support batch import.",
            prd_contract_version="v1",
            knowledge_base_snapshot="kb-1",
            repository="example/service",
            repository_commit="abc123",
        )

    def passage(
        self,
        *,
        chunk_id: str = "chunk-1",
        score: float | None = 0.8,
        project_id: str = "project-1",
        source_type: SourceType = SourceType.DOCUMENT,
        repository_commit: str | None = None,
    ) -> GatewayPassage:
        return GatewayPassage(
            chunk_id=chunk_id,
            content="Existing import behavior.",
            source_type=source_type,
            source_uri="s3://example/source",
            locator="Import workflow",
            project_id=project_id,
            retrieval_score=score,
            repository_commit=repository_commit,
        )

    async def test_deduplicates_queries_and_chunks_using_highest_score(self) -> None:
        low = self.passage(score=0.4)
        high = self.passage(score=0.9)
        second = self.passage(chunk_id="chunk-2", score=0.7)
        gateway = FakeGateway({"imports": [low, second], "batch": [high]})
        manifest = FakeManifestProvider()
        retriever = AgentCoreKnowledgeBaseRetriever(
            gateway=gateway,
            manifest_provider=manifest,
            user_id="user-1",
            top_k=7,
        )
        result = await retriever.retrieve(
            self.request(), [" imports ", "batch", "imports"]
        )
        self.assertEqual([chunk.chunk_id for chunk in result.chunks], ["chunk-1", "chunk-2"])
        self.assertEqual(result.chunks[0].retrieval_score, 0.9)
        self.assertEqual([call["query"] for call in gateway.calls], ["imports", "batch"])
        self.assertTrue(all(call["project_id"] == "project-1" for call in gateway.calls))
        self.assertTrue(all(call["user_id"] == "user-1" for call in gateway.calls))
        self.assertTrue(all(call["top_k"] == 7 for call in gateway.calls))
        self.assertEqual(
            result.source_manifest["src/imports.py"],
            frozenset({"ImportService", "ImportService.create"}),
        )

    async def test_cross_project_passage_fails_closed(self) -> None:
        gateway = FakeGateway({"imports": [self.passage(project_id="project-2")]})
        retriever = AgentCoreKnowledgeBaseRetriever(
            gateway=gateway,
            manifest_provider=FakeManifestProvider(),
            user_id="user-1",
        )
        with self.assertRaisesRegex(KnowledgeBaseRetrievalError, "cross-project"):
            await retriever.retrieve(self.request(), ["imports"])

    async def test_code_passage_must_match_pinned_commit(self) -> None:
        gateway = FakeGateway(
            {
                "imports": [
                    self.passage(
                        source_type=SourceType.CODE,
                        repository_commit="wrong-commit",
                    )
                ]
            }
        )
        retriever = AgentCoreKnowledgeBaseRetriever(
            gateway=gateway,
            manifest_provider=FakeManifestProvider(),
            user_id="user-1",
        )
        with self.assertRaisesRegex(KnowledgeBaseRetrievalError, "pinned"):
            await retriever.retrieve(self.request(), ["imports"])

    async def test_valid_code_passage_is_returned(self) -> None:
        gateway = FakeGateway(
            {
                "imports": [
                    self.passage(
                        source_type=SourceType.CODE,
                        repository_commit="abc123",
                    )
                ]
            }
        )
        result = await AgentCoreKnowledgeBaseRetriever(
            gateway=gateway,
            manifest_provider=FakeManifestProvider(),
            user_id="user-1",
        ).retrieve(self.request(), ["imports"])
        self.assertEqual(result.chunks[0].repository_commit, "abc123")

    async def test_empty_results_and_queries_fail_closed(self) -> None:
        retriever = AgentCoreKnowledgeBaseRetriever(
            gateway=FakeGateway({}),
            manifest_provider=FakeManifestProvider(),
            user_id="user-1",
        )
        with self.assertRaisesRegex(KnowledgeBaseRetrievalError, "no passages"):
            await retriever.retrieve(self.request(), ["imports"])
        with self.assertRaisesRegex(KnowledgeBaseRetrievalError, "non-empty query"):
            await retriever.retrieve(self.request(), [" ", ""])

    async def test_query_budget_is_enforced_before_gateway_call(self) -> None:
        gateway = FakeGateway({})
        retriever = AgentCoreKnowledgeBaseRetriever(
            gateway=gateway,
            manifest_provider=FakeManifestProvider(),
            user_id="user-1",
            max_queries=2,
        )
        with self.assertRaisesRegex(KnowledgeBaseRetrievalError, "maximum of 2"):
            await retriever.retrieve(self.request(), ["one", "two", "three"])
        self.assertEqual(gateway.calls, [])

    async def test_non_code_passage_cannot_carry_commit(self) -> None:
        gateway = FakeGateway(
            {"imports": [self.passage(repository_commit="abc123")]}
        )
        retriever = AgentCoreKnowledgeBaseRetriever(
            gateway=gateway,
            manifest_provider=FakeManifestProvider(),
            user_id="user-1",
        )
        with self.assertRaisesRegex(KnowledgeBaseRetrievalError, "non-code"):
            await retriever.retrieve(self.request(), ["imports"])


if __name__ == "__main__":
    unittest.main()
