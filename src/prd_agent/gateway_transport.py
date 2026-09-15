"""Concrete Strands MCP transport for AgentCore Knowledge Base Gateway."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any, Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator
from strands.tools.mcp.mcp_client import MCPClient

from prd_agent.contracts import SourceType, StrictModel
from prd_agent.knowledge_base import GatewayPassage, KnowledgeBaseRetrievalError


class MCPClientLike(Protocol):
    def __enter__(self) -> MCPClientLike: ...

    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

    def call_tool_sync(
        self,
        tool_use_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
    ) -> Mapping[str, Any]: ...


class AgentCoreGatewayConfig(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    retrieve_tool_name: str = Field(min_length=1)
    timeout_seconds: int = Field(default=60, ge=1, le=900)
    search_type: str = "HYBRID"
    send_managed_search_configuration: bool = False

    @field_validator("url")
    @classmethod
    def require_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("AgentCore Gateway URL must use HTTPS")
        return value

    @field_validator("search_type")
    @classmethod
    def validate_search_type(cls, value: str) -> str:
        if value not in {"HYBRID", "SEMANTIC"}:
            raise ValueError("search_type must be HYBRID or SEMANTIC")
        return value


ClientFactory = Callable[[dict[str, str]], MCPClientLike]
TokenProvider = Callable[[], str]


class MCPGatewayRetrieveClient:
    """Invoke the AgentCore managed-KB Retrieve tool through Strands MCPClient."""

    def __init__(
        self,
        *,
        config: AgentCoreGatewayConfig,
        access_token_provider: TokenProvider | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        if access_token_provider is None and client_factory is None:
            raise ValueError(
                "access_token_provider or an authenticated client_factory is required"
            )
        self._config = config
        self._access_token_provider = access_token_provider
        self._client_factory = client_factory or self._default_client_factory

    def _default_client_factory(self, headers: dict[str, str]) -> MCPClient:
        return MCPClient(
            url=self._config.url,
            headers=headers,
            application_name="agentcore-prd-agent",
            continue_on_error=False,
        )

    async def retrieve(
        self,
        *,
        query: str,
        project_id: str,
        user_id: str,
        top_k: int,
    ) -> Sequence[GatewayPassage]:
        if not query.strip() or not project_id or not user_id:
            raise ValueError("query, project_id, and user_id are required")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        headers: dict[str, str] = {}
        if self._access_token_provider is not None:
            token = self._access_token_provider()
            if not token:
                raise KnowledgeBaseRetrievalError(
                    "AgentCore Gateway access token is unavailable"
                )
            headers["Authorization"] = f"Bearer {token}"

        arguments: dict[str, Any] = {
            "retrievalQuery": {"text": query.strip()},
            "userContext": {"userId": user_id},
        }
        if self._config.send_managed_search_configuration:
            arguments["retrievalConfiguration"] = {
                "managedSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": self._config.search_type,
                    "filter": {
                        "equals": {
                            "key": "project_id",
                            "value": project_id,
                        }
                    },
                }
            }
        result = await asyncio.to_thread(
            self._call_tool,
            headers,
            arguments,
        )
        return self._parse_result(result)

    def _call_tool(
        self,
        headers: dict[str, str],
        arguments: dict[str, Any],
    ) -> Mapping[str, Any]:
        with self._client_factory(headers) as client:
            return client.call_tool_sync(
                uuid4().hex,
                self._config.retrieve_tool_name,
                arguments,
                timedelta(seconds=self._config.timeout_seconds),
            )

    @staticmethod
    def _parse_result(result: Mapping[str, Any]) -> list[GatewayPassage]:
        if result.get("status") != "success" or result.get("isError") is True:
            raise KnowledgeBaseRetrievalError("AgentCore Gateway Retrieve tool failed")
        payload = result.get("structuredContent")
        if payload is None:
            content = result.get("content")
            if not isinstance(content, list):
                raise KnowledgeBaseRetrievalError(
                    "AgentCore Gateway Retrieve returned no content list"
                )
            text_payload = next(
                (
                    block.get("text")
                    for block in content
                    if isinstance(block, Mapping)
                    and isinstance(block.get("text"), str)
                ),
                None,
            )
            if text_payload is None:
                raise KnowledgeBaseRetrievalError(
                    "AgentCore Gateway Retrieve returned no text payload"
                )
            try:
                import json

                payload = json.loads(text_payload)
            except (TypeError, json.JSONDecodeError) as error:
                raise KnowledgeBaseRetrievalError(
                    "AgentCore Gateway Retrieve returned invalid JSON text"
                ) from error
        if not isinstance(payload, Mapping):
            raise KnowledgeBaseRetrievalError(
                "AgentCore Gateway Retrieve payload must be an object"
            )
        raw_results = payload.get("retrievalResults")
        if not isinstance(raw_results, list):
            raise KnowledgeBaseRetrievalError(
                "AgentCore Gateway Retrieve response has no retrievalResults list"
            )
        return [MCPGatewayRetrieveClient._parse_passage(item) for item in raw_results]

    @staticmethod
    def _parse_passage(item: Any) -> GatewayPassage:
        if not isinstance(item, Mapping):
            raise KnowledgeBaseRetrievalError("Knowledge Base passage must be an object")
        metadata = item.get("metadata")
        content_block = item.get("content")
        if not isinstance(metadata, Mapping) or not isinstance(content_block, Mapping):
            raise KnowledgeBaseRetrievalError(
                "Knowledge Base passage requires content and metadata objects"
            )

        text = content_block.get("text")
        chunk_id = metadata.get("chunk_id")
        project_id = metadata.get("project_id")
        source_type_value = metadata.get("source_type")
        source_uri = metadata.get("source_uri") or _location_uri(item.get("location"))
        locator = metadata.get("locator") or _metadata_locator(metadata)
        if not all(
            isinstance(value, str) and value
            for value in (text, chunk_id, project_id, source_type_value, source_uri, locator)
        ):
            raise KnowledgeBaseRetrievalError(
                "Knowledge Base passage is missing required traceability metadata"
            )

        try:
            source_type = SourceType(source_type_value)
        except ValueError as error:
            raise KnowledgeBaseRetrievalError(
                "Knowledge Base passage has an invalid source_type"
            ) from error

        score = item.get("score")
        if score is not None and not isinstance(score, (int, float)):
            raise KnowledgeBaseRetrievalError("Knowledge Base passage score must be numeric")
        repository_commit = metadata.get("commit_sha")
        if repository_commit is not None and not isinstance(repository_commit, str):
            raise KnowledgeBaseRetrievalError("commit_sha metadata must be a string")

        return GatewayPassage(
            chunk_id=chunk_id,
            content=text,
            source_type=source_type,
            source_uri=source_uri,
            locator=locator,
            project_id=project_id,
            retrieval_score=float(score) if score is not None else None,
            repository_commit=repository_commit,
        )


def _location_uri(location: Any) -> str | None:
    if not isinstance(location, Mapping):
        return None
    for key, field in (
        ("s3Location", "uri"),
        ("webLocation", "url"),
        ("confluenceLocation", "url"),
        ("sharePointLocation", "url"),
    ):
        value = location.get(key)
        if isinstance(value, Mapping) and isinstance(value.get(field), str):
            return value[field]
    return None


def _metadata_locator(metadata: Mapping[str, Any]) -> str | None:
    heading = metadata.get("heading")
    if isinstance(heading, str) and heading:
        return heading
    file_path = metadata.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None
    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    if isinstance(line_start, int) and isinstance(line_end, int):
        return f"{file_path}:{line_start}-{line_end}"
    return file_path
