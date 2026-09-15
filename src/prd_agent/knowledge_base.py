"""Strict AgentCore Knowledge Base retrieval boundary.

The concrete MCP/HTTP transport is deliberately injected. This module owns the
security and data-integrity rules that must remain independent of transport.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import ConfigDict, Field

from prd_agent.contracts import GenerationRequest, SourceType, StrictModel
from prd_agent.workflow import RetrievalBundle, RetrievedChunk


class KnowledgeBaseRetrievalError(RuntimeError):
    """Raised when Gateway results cannot safely support PRD generation."""


class GatewayPassage(StrictModel):
    """Normalized passage returned by the AgentCore KB Gateway transport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_type: SourceType
    source_uri: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    retrieval_score: float | None = None
    repository_commit: str | None = None


class GatewayRetrieveClient(Protocol):
    async def retrieve(
        self,
        *,
        query: str,
        project_id: str,
        user_id: str,
        top_k: int,
    ) -> Sequence[GatewayPassage]: ...


class SourceManifestProvider(Protocol):
    async def load(
        self,
        *,
        project_id: str,
        repository: str,
        commit_sha: str,
    ) -> Mapping[str, Sequence[str]]: ...


class AgentCoreKnowledgeBaseRetriever:
    """The workflow's only retrieval implementation.

    Project and user context are supplied by trusted application state, never by
    the model. Results crossing a project or repository version boundary fail
    closed before reaching a Strands prompt.
    """

    def __init__(
        self,
        *,
        gateway: GatewayRetrieveClient,
        manifest_provider: SourceManifestProvider,
        user_id: str,
        top_k: int = 10,
        max_queries: int = 12,
        allow_empty: bool = False,
    ) -> None:
        if not user_id:
            raise ValueError("user_id is required")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        self._gateway = gateway
        self._manifest_provider = manifest_provider
        self._user_id = user_id
        self._top_k = top_k
        self._max_queries = max_queries
        self._allow_empty = allow_empty

    async def retrieve(
        self,
        request: GenerationRequest,
        queries: Sequence[str],
    ) -> RetrievalBundle:
        normalized_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not normalized_queries:
            raise KnowledgeBaseRetrievalError("at least one non-empty query is required")
        if len(normalized_queries) > self._max_queries:
            raise KnowledgeBaseRetrievalError(
                f"query count exceeds configured maximum of {self._max_queries}"
            )

        passage_groups, manifest = await asyncio.gather(
            asyncio.gather(
                *(
                    self._gateway.retrieve(
                        query=query,
                        project_id=request.project_id,
                        user_id=self._user_id,
                        top_k=self._top_k,
                    )
                    for query in normalized_queries
                )
            ),
            self._manifest_provider.load(
                project_id=request.project_id,
                repository=request.repository,
                commit_sha=request.repository_commit,
            ),
        )

        by_chunk_id: dict[str, GatewayPassage] = {}
        for passages in passage_groups:
            for passage in passages:
                self._validate_passage(request, passage)
                existing = by_chunk_id.get(passage.chunk_id)
                if existing is None or self._score(passage) > self._score(existing):
                    by_chunk_id[passage.chunk_id] = passage

        if not by_chunk_id and not self._allow_empty:
            raise KnowledgeBaseRetrievalError(
                "AgentCore Knowledge Base returned no passages for this request"
            )

        ordered = sorted(
            by_chunk_id.values(),
            key=lambda passage: (
                -self._score(passage),
                passage.source_uri,
                passage.locator,
                passage.chunk_id,
            ),
        )
        return RetrievalBundle(
            chunks=[
                RetrievedChunk(
                    chunk_id=passage.chunk_id,
                    source_type=passage.source_type,
                    source_uri=passage.source_uri,
                    locator=passage.locator,
                    content=passage.content,
                    retrieval_score=passage.retrieval_score,
                    repository_commit=passage.repository_commit,
                )
                for passage in ordered
            ],
            source_manifest={
                path: frozenset(symbols) for path, symbols in manifest.items()
            },
        )

    @staticmethod
    def _score(passage: GatewayPassage) -> float:
        return passage.retrieval_score if passage.retrieval_score is not None else -1.0

    @staticmethod
    def _validate_passage(
        request: GenerationRequest,
        passage: GatewayPassage,
    ) -> None:
        if passage.project_id != request.project_id:
            raise KnowledgeBaseRetrievalError(
                "AgentCore Knowledge Base returned a cross-project passage"
            )
        if passage.source_type is SourceType.CODE:
            if passage.repository_commit != request.repository_commit:
                raise KnowledgeBaseRetrievalError(
                    "code passage does not match the pinned repository commit"
                )
        elif passage.repository_commit is not None:
            raise KnowledgeBaseRetrievalError(
                "non-code passage unexpectedly contains a repository commit"
            )
