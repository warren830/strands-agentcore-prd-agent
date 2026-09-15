from __future__ import annotations

import asyncio
import unittest

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from prd_agent.contract import PrdContract, SectionContract, SectionKind
from prd_agent.contracts import (
    GenerationRequest,
    GenerationResult,
    RunStatus,
    VersionSet,
)
from prd_agent.runtime import (
    PrdGenerationService,
    StaticContractProvider,
    StaticVersionProvider,
    create_runtime_app,
    stream_runtime_invocation,
)


class FakeWorkflow:
    def __init__(self, result: GenerationResult | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


async def collect_events(iterator):
    return [event async for event in iterator]


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    def contract(self) -> PrdContract:
        return PrdContract(
            version="v1",
            document_title="Product Requirements Document",
            sections=[
                SectionContract(
                    section_id="summary",
                    title="1. Summary",
                    kind=SectionKind.PARAGRAPH,
                )
            ],
        )

    def versions(self) -> VersionSet:
        return VersionSet(
            knowledge_base_snapshot="kb-1",
            repository_commit="abc123",
            prd_contract_version="v1",
            prompt_version="prompt-1",
            runtime_version="runtime-1",
            model_id="us.anthropic.claude-opus-4-6-v1",
        )

    def payload(self):
        return {
            "run_id": "run-1",
            "project_id": "project-1",
            "requirement_text": "Support batch import.",
            "prd_contract_version": "v1",
            "knowledge_base_snapshot": "kb-1",
            "repository": "example/service",
            "repository_commit": "abc123",
        }

    def service(self, workflow: FakeWorkflow, contract: PrdContract | None = None):
        return PrdGenerationService(
            workflow=workflow,  # type: ignore[arg-type]
            contract_provider=StaticContractProvider(contract or self.contract()),
            version_provider=StaticVersionProvider(
                prompt_version="prompt-1",
                runtime_version="runtime-1",
                model_id="us.anthropic.claude-opus-4-6-v1",
            ),
        )

    async def test_valid_request_streams_started_and_result(self) -> None:
        result = GenerationResult(
            run_id="run-1",
            status=RunStatus.COMPLETED,
            versions=self.versions(),
            prd_markdown="# Product Requirements Document\n",
        )
        workflow = FakeWorkflow(result=result)
        events = await collect_events(
            stream_runtime_invocation(self.service(workflow), self.payload())
        )
        self.assertEqual([event["event"] for event in events], ["started", "result"])
        self.assertEqual(events[1]["run_id"], "run-1")
        self.assertEqual(events[1]["result"]["status"], "completed")
        self.assertEqual(workflow.calls[0]["contract"].version, "v1")
        self.assertEqual(workflow.calls[0]["versions"], self.versions())

    async def test_invalid_payload_fails_before_workflow(self) -> None:
        workflow = FakeWorkflow()
        events = await collect_events(
            stream_runtime_invocation(self.service(workflow), {"run_id": "run-1"})
        )
        self.assertEqual(events[0]["event"], "error")
        self.assertEqual(events[0]["error"]["code"], "INVALID_REQUEST")
        self.assertGreater(len(events[0]["error"]["details"]), 0)
        self.assertEqual(workflow.calls, [])

    async def test_non_object_payload_is_rejected(self) -> None:
        workflow = FakeWorkflow()
        events = await collect_events(
            stream_runtime_invocation(self.service(workflow), "untrusted string")
        )
        self.assertEqual(events[0]["error"]["code"], "INVALID_REQUEST")
        self.assertNotIn("run_id", events[0])

    async def test_missing_contract_has_stable_error(self) -> None:
        payload = self.payload()
        payload["prd_contract_version"] = "v2"
        events = await collect_events(
            stream_runtime_invocation(self.service(FakeWorkflow()), payload)
        )
        self.assertEqual([event["event"] for event in events], ["started", "error"])
        self.assertEqual(events[1]["error"]["code"], "CONFIGURATION_NOT_FOUND")

    async def test_internal_exception_is_not_leaked(self) -> None:
        workflow = FakeWorkflow(error=RuntimeError("secret internal detail"))
        events = await collect_events(
            stream_runtime_invocation(self.service(workflow), self.payload())
        )
        self.assertEqual(events[1]["error"]["code"], "GENERATION_FAILED")
        self.assertNotIn("secret internal detail", str(events[1]))
        self.assertIn("run ID", events[1]["error"]["message"])

    async def test_long_generation_streams_progress_heartbeats(self) -> None:
        result = GenerationResult(
            run_id="run-1",
            status=RunStatus.COMPLETED,
            versions=self.versions(),
            prd_markdown="# Product Requirements Document\n",
        )

        class SlowService:
            async def generate(self, **kwargs):
                await asyncio.sleep(0.025)
                return result

        events = await collect_events(
            stream_runtime_invocation(
                SlowService(),  # type: ignore[arg-type]
                self.payload(),
                heartbeat_seconds=0.005,
            )
        )
        event_names = [event["event"] for event in events]
        self.assertEqual(event_names[0], "started")
        self.assertEqual(event_names[-1], "result")
        self.assertGreaterEqual(event_names.count("progress"), 2)

    def test_agentcore_app_registers_invocations_and_ping_routes(self) -> None:
        result = GenerationResult(
            run_id="run-1",
            status=RunStatus.COMPLETED,
            versions=self.versions(),
            prd_markdown="# Product Requirements Document\n",
        )
        app = create_runtime_app(self.service(FakeWorkflow(result=result)))
        self.assertIsInstance(app, BedrockAgentCoreApp)
        paths = {getattr(route, "path", None) for route in app.routes}
        self.assertIn("/invocations", paths)
        self.assertIn("/ping", paths)

    async def test_static_version_provider_uses_request_snapshot(self) -> None:
        request = GenerationRequest.model_validate(
            {key: value for key, value in self.payload().items() if key != "run_id"}
        )
        provider = StaticVersionProvider(
            prompt_version="prompt-2",
            runtime_version="runtime-3",
            model_id="global.anthropic.claude-opus-4-6-v1",
        )
        versions = await provider.resolve(request)
        self.assertEqual(versions.knowledge_base_snapshot, "kb-1")
        self.assertEqual(versions.repository_commit, "abc123")
        self.assertEqual(versions.prompt_version, "prompt-2")


if __name__ == "__main__":
    unittest.main()
