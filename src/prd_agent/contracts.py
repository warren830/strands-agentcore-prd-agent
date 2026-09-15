"""Core typed contracts for the PRD generation workflow.

These models form the stable boundary between probabilistic model output and
fully deterministic validation/rendering. Project-specific PRD schemas will be
compiled separately from each customer's authoritative template.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonEmptyText = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields at every workflow boundary."""

    model_config = ConfigDict(extra="forbid")


class SourceType(StrEnum):
    DOCUMENT = "document"
    CODE = "code"
    USER = "user"


class EvidenceStatus(StrEnum):
    CONFIRMED = "confirmed"
    ASSUMPTION = "assumption"
    UNRESOLVED = "unresolved"


class RunStatus(StrEnum):
    CREATED = "created"
    INGESTING = "ingesting"
    READY = "ready"
    ANALYZING = "analyzing"
    NEEDS_CLARIFICATION = "needs_clarification"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    INGESTION_FAILED = "ingestion_failed"
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"
    FAILED_VALIDATION = "failed_validation"
    EVALUATION_UNAVAILABLE = "evaluation_unavailable"


class Citation(StrictModel):
    """A reference to evidence returned by AgentCore Knowledge Base."""

    chunk_id: NonEmptyText
    source_type: SourceType
    source_uri: NonEmptyText
    locator: NonEmptyText
    retrieved_at: NonEmptyText
    retrieval_score: float | None = None
    repository_commit: str | None = None


class EvidenceBackedText(StrictModel):
    """A claim whose evidence state is explicit and mechanically enforceable."""

    text: NonEmptyText
    status: EvidenceStatus
    citations: list[Citation] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def normalize_null_citations(cls, value):
        return [] if value is None else value

    @model_validator(mode="after")
    def require_citations_for_confirmed_claims(self) -> EvidenceBackedText:
        if self.status is EvidenceStatus.CONFIRMED and not self.citations:
            raise ValueError("confirmed claims require at least one citation")
        return self


class ClarificationQuestion(StrictModel):
    """A bounded question linked to fields that cannot yet be generated safely."""

    question_id: NonEmptyText
    question: NonEmptyText
    rationale: NonEmptyText
    blocked_fields: list[NonEmptyText] = Field(min_length=1)


class RequirementIntent(StrictModel):
    """Normalized English interpretation of the user's original requirement."""

    original_requirement: NonEmptyText
    normalized_english_requirement: NonEmptyText
    problem_statement: NonEmptyText
    target_users: list[NonEmptyText] = Field(default_factory=list)
    desired_outcomes: list[NonEmptyText] = Field(default_factory=list)
    explicit_constraints: list[NonEmptyText] = Field(default_factory=list)
    assumptions: list[NonEmptyText] = Field(default_factory=list)
    search_queries: list[NonEmptyText] = Field(min_length=1)
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_clarification_budget(self) -> RequirementIntent:
        if len(self.clarification_questions) > 3:
            raise ValueError("at most three clarification questions are allowed")
        return self


class ImpactType(StrEnum):
    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"
    CONFIGURATION = "configuration"
    TEST_ONLY = "test_only"


class CodeImpact(StrictModel):
    """A source-code impact anchored to an immutable repository commit."""

    repository: NonEmptyText
    commit_sha: NonEmptyText
    file_path: NonEmptyText
    symbol: str | None = None
    impact_type: ImpactType
    rationale: EvidenceBackedText


class VersionSet(StrictModel):
    """Versions required to reproduce and audit a generated PRD."""

    knowledge_base_snapshot: NonEmptyText
    repository_commit: NonEmptyText
    prd_contract_version: NonEmptyText
    prompt_version: NonEmptyText
    runtime_version: NonEmptyText
    model_id: Literal[
        "anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-opus-4-6-v1",
        "eu.anthropic.claude-opus-4-6-v1",
        "au.anthropic.claude-opus-4-6-v1",
        "global.anthropic.claude-opus-4-6-v1",
    ]


class GenerationRequest(StrictModel):
    """Input accepted by the deterministic PRD workflow orchestrator."""

    project_id: NonEmptyText
    requirement_text: NonEmptyText
    prd_contract_version: NonEmptyText
    knowledge_base_snapshot: NonEmptyText
    repository: NonEmptyText
    repository_commit: NonEmptyText
    mode: Literal["interactive", "one_shot"] = "interactive"
    output_language: Literal["en-US"] = "en-US"


class ValidationIssue(StrictModel):
    """A deterministic validation failure suitable for bounded repair."""

    rule_id: NonEmptyText
    field_path: NonEmptyText
    message: NonEmptyText


class ValidationReport(StrictModel):
    """Content-minimized evidence and deterministic validation sidecar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    repair_attempts: int = Field(ge=0, le=2)
    retrieved_chunk_ids: tuple[str, ...]
    claim_count: int = Field(ge=0)
    confirmed_claim_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    code_impact_count: int = Field(ge=0)
    citations: tuple[Citation, ...]
    code_impacts: tuple[CodeImpact, ...]
    issues: tuple[ValidationIssue, ...]

    @model_validator(mode="after")
    def validate_pass_state(self) -> ValidationReport:
        if self.passed and self.issues:
            raise ValueError("a passed report cannot contain validation issues")
        if not self.passed and not self.issues:
            raise ValueError("a failed report requires validation issues")
        if self.confirmed_claim_count > self.claim_count:
            raise ValueError("confirmed_claim_count cannot exceed claim_count")
        if self.citation_count != len(self.citations):
            raise ValueError("citation_count must match unique citations")
        if self.code_impact_count != len(self.code_impacts):
            raise ValueError("code_impact_count must match code impacts")
        if len(self.retrieved_chunk_ids) != len(set(self.retrieved_chunk_ids)):
            raise ValueError("retrieved chunk IDs must be unique")
        return self


class GenerationResult(StrictModel):
    """Stable response envelope for all successful and failed workflow states."""

    run_id: NonEmptyText
    status: RunStatus
    versions: VersionSet | None = None
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    prd_markdown: str | None = None
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    validation_report: ValidationReport | None = None

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> GenerationResult:
        if self.status is RunStatus.COMPLETED:
            if not self.prd_markdown or self.versions is None:
                raise ValueError("completed results require PRD content and versions")
            if self.validation_report is not None and not self.validation_report.passed:
                raise ValueError("completed results cannot contain a failed validation report")
        elif self.prd_markdown is not None:
            raise ValueError("only completed results may contain final PRD content")

        if self.status is RunStatus.NEEDS_CLARIFICATION:
            if not self.clarification_questions:
                raise ValueError("clarification state requires at least one question")
            if len(self.clarification_questions) > 3:
                raise ValueError("at most three clarification questions are allowed")

        if self.status is RunStatus.FAILED_VALIDATION:
            if not self.validation_issues:
                raise ValueError("failed validation requires at least one issue")
            if self.validation_report is not None and self.validation_report.passed:
                raise ValueError("failed validation cannot contain a passed report")
        return self
