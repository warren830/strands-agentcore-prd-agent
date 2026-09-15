"""Pure deployment plans and least-privilege policy builders."""

from __future__ import annotations

import re
from typing import Any, Self

from pydantic import ConfigDict, Field, model_validator

from prd_agent.contracts import StrictModel

_PREFIX = re.compile(r"^[a-z][a-z0-9-]{2,30}$")


class DeploymentPlan(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account_id: str = Field(pattern=r"^\d{12}$")
    region: str = Field(min_length=1)
    prefix: str
    project_id: str = "mock-prd-e2e"
    repository: str = "agentcore-prd-agent"

    @model_validator(mode="after")
    def validate_prefix(self) -> Self:
        if not _PREFIX.fullmatch(self.prefix):
            raise ValueError("prefix must be lowercase alphanumeric with hyphens")
        return self

    @property
    def bucket_name(self) -> str:
        return f"{self.prefix}-{self.account_id}-{self.region}"

    @property
    def knowledge_base_name(self) -> str:
        return f"{self.prefix}-kb"

    @property
    def data_source_name(self) -> str:
        return f"{self.prefix}-s3"

    @property
    def gateway_name(self) -> str:
        return f"{self.prefix}-gw"

    @property
    def gateway_target_name(self) -> str:
        return f"{self.prefix}-kb"

    @property
    def runtime_name(self) -> str:
        return f"{self.prefix.replace('-', '_')}_direct"

    @property
    def kb_role_name(self) -> str:
        return "PrdAgentMockE2EKnowledgeBaseRole"

    @property
    def gateway_role_name(self) -> str:
        return "PrdAgentMockE2EGatewayRole"

    @property
    def runtime_role_name(self) -> str:
        return "PrdAgentMockE2ERuntimeRole"

    @property
    def tags(self) -> dict[str, str]:
        return {
            "Project": "AgentCorePrdAgent",
            "Environment": "test-in-production-account",
            "ManagedBy": "prd-agent-provisioner",
            "Purpose": "mock-e2e",
        }

    def summary(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "region": self.region,
            "prefix": self.prefix,
            "bucket_name": self.bucket_name,
            "knowledge_base_name": self.knowledge_base_name,
            "data_source_name": self.data_source_name,
            "gateway_name": self.gateway_name,
            "gateway_target_name": self.gateway_target_name,
            "runtime_name": self.runtime_name,
            "roles": {
                "knowledge_base": self.kb_role_name,
                "gateway": self.gateway_role_name,
                "runtime": self.runtime_role_name,
            },
            "model_id": "us.anthropic.claude-opus-4-6-v1",
            "deletion_operations": [],
            "tags": self.tags,
        }


def _trust_policy(
    *,
    service: str,
    account_id: str,
    source_arn: str,
) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": service},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account_id},
                    "ArnLike": {"aws:SourceArn": source_arn},
                },
            }
        ],
    }


def knowledge_base_trust(plan: DeploymentPlan) -> dict[str, Any]:
    return _trust_policy(
        service="bedrock.amazonaws.com",
        account_id=plan.account_id,
        source_arn=(
            f"arn:aws:bedrock:{plan.region}:{plan.account_id}:knowledge-base/*"
        ),
    )


def knowledge_base_permissions(plan: DeploymentPlan) -> dict[str, Any]:
    bucket_arn = f"arn:aws:s3:::{plan.bucket_name}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListDedicatedSourceBucket",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": bucket_arn,
                "Condition": {
                    "StringEquals": {"aws:ResourceAccount": plan.account_id}
                },
            },
            {
                "Sid": "ReadDedicatedSourceObjects",
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"{bucket_arn}/content/*",
                "Condition": {
                    "StringEquals": {"aws:ResourceAccount": plan.account_id}
                },
            },
        ],
    }


def gateway_trust(plan: DeploymentPlan) -> dict[str, Any]:
    return _trust_policy(
        service="bedrock-agentcore.amazonaws.com",
        account_id=plan.account_id,
        source_arn=(
            f"arn:aws:bedrock-agentcore:{plan.region}:{plan.account_id}:gateway/*"
        ),
    )


def gateway_permissions(plan: DeploymentPlan, knowledge_base_id: str) -> dict[str, Any]:
    kb_arn = (
        f"arn:aws:bedrock:{plan.region}:{plan.account_id}:"
        f"knowledge-base/{knowledge_base_id}"
    )
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ValidateAndRetrieveManagedKnowledgeBase",
                "Effect": "Allow",
                "Action": ["bedrock:GetKnowledgeBase", "bedrock:Retrieve"],
                "Resource": kb_arn,
            }
        ],
    }


def runtime_trust(plan: DeploymentPlan) -> dict[str, Any]:
    return _trust_policy(
        service="bedrock-agentcore.amazonaws.com",
        account_id=plan.account_id,
        source_arn=(
            f"arn:aws:bedrock-agentcore:{plan.region}:{plan.account_id}:runtime/*"
        ),
    )


def runtime_permissions(plan: DeploymentPlan, gateway_id: str) -> dict[str, Any]:
    model_id = "anthropic.claude-opus-4-6-v1"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeClaudeOpus46",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                "Resource": [
                    *[
                        f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
                        for region in ("us-east-1", "us-east-2", "us-west-2")
                    ],
                    (
                        f"arn:aws:bedrock:{plan.region}:{plan.account_id}:"
                        "inference-profile/us.anthropic.claude-opus-4-6-v1"
                    ),
                ],
            },
            {
                "Sid": "InvokeDedicatedGateway",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeGateway",
                "Resource": (
                    f"arn:aws:bedrock-agentcore:{plan.region}:{plan.account_id}:"
                    f"gateway/{gateway_id}"
                ),
            },
            {
                "Sid": "RuntimeLogGroups",
                "Effect": "Allow",
                "Action": ["logs:DescribeLogStreams", "logs:CreateLogGroup"],
                "Resource": (
                    f"arn:aws:logs:{plan.region}:{plan.account_id}:"
                    "log-group:/aws/bedrock-agentcore/runtimes/*"
                ),
            },
            {
                "Sid": "RuntimeLogResourcePolicy",
                "Effect": "Allow",
                "Action": "logs:PutResourcePolicy",
                "Resource": (
                    f"arn:aws:logs:{plan.region}:{plan.account_id}:"
                    f"log-group:/aws/bedrock-agentcore/runtimes/{plan.runtime_name}-*"
                ),
            },
            {
                "Sid": "DescribeRuntimeLogGroups",
                "Effect": "Allow",
                "Action": "logs:DescribeLogGroups",
                "Resource": (
                    f"arn:aws:logs:{plan.region}:{plan.account_id}:log-group:*"
                ),
            },
            {
                "Sid": "WriteRuntimeLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": (
                    f"arn:aws:logs:{plan.region}:{plan.account_id}:"
                    "log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
                ),
            },
            {
                "Sid": "WriteRuntimeTraces",
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                "Resource": "*",
            },
            {
                "Sid": "WriteAgentCoreMetrics",
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}
                },
            },
        ],
    }


def managed_knowledge_base_configuration() -> dict[str, Any]:
    return {
        "type": "MANAGED",
        "managedKnowledgeBaseConfiguration": {"embeddingModelType": "MANAGED"},
    }


def managed_s3_data_source_configuration(plan: DeploymentPlan) -> dict[str, Any]:
    return {
        "type": "MANAGED_KNOWLEDGE_BASE_CONNECTOR",
        "managedKnowledgeBaseConnectorConfiguration": {
            "connectorParameters": {
                "type": "S3",
                "version": "1",
                "connectionConfiguration": {
                    "bucketName": plan.bucket_name,
                    "bucketOwnerAccountId": plan.account_id,
                },
                "filterConfiguration": {
                    "inclusionPrefixes": ["content/"],
                },
            },
            "deletionProtectionConfiguration": {
                "deletionProtectionStatus": "ENABLED",
                "deletionProtectionThreshold": 20,
            },
        },
    }


def gateway_target_configuration(
    plan: DeploymentPlan,
    knowledge_base_id: str,
    source_snapshot: str,
) -> dict[str, Any]:
    version_filter = {
        "orAll": [
            {
                "andAll": [
                    {"equals": {"key": "project_id", "value": plan.project_id}},
                    {"equals": {"key": "source_type", "value": "document"}},
                ]
            },
            {
                "andAll": [
                    {"equals": {"key": "project_id", "value": plan.project_id}},
                    {"equals": {"key": "source_type", "value": "code"}},
                    {
                        "equals": {
                            "key": "commit_sha",
                            "value": source_snapshot,
                        }
                    },
                ]
            },
        ]
    }
    return {
        "mcp": {
            "connector": {
                "source": {"connectorId": "bedrock-knowledge-bases"},
                "configurations": [
                    {
                        "name": "Retrieve",
                        "description": (
                            "Retrieve evidence for English PRD generation from the "
                            "dedicated mock knowledge base."
                        ),
                        "parameterValues": {
                            "knowledgeBaseId": knowledge_base_id,
                            "retrievalConfiguration": {
                                "managedSearchConfiguration": {
                                    "numberOfResults": 10,
                                    "overrideSearchType": "HYBRID",
                                    "filter": version_filter,
                                }
                            },
                        },
                        "parameterOverrides": [
                            {
                                "path": "$.retrievalQuery.text",
                                "description": "Specific evidence search query.",
                                "visible": True,
                            },
                            {
                                "path": "$.userContext",
                                "description": "Trusted caller context.",
                                "visible": True,
                            },
                        ],
                    }
                ],
            }
        }
    }
