from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from prd_agent.application import JsonContractProvider, build_runtime_app
from prd_agent.contract import PrdContract, SectionContract, SectionKind
from prd_agent.settings import AppSettings, DeploymentEnvironment


class ApplicationTest(unittest.IsolatedAsyncioTestCase):
    def contract(self) -> PrdContract:
        return PrdContract(
            version="mock-v1",
            document_title="Mock PRD",
            sections=[
                SectionContract(
                    section_id="summary",
                    title="1. Summary",
                    kind=SectionKind.PARAGRAPH,
                )
            ],
        )

    async def test_json_contract_provider_enforces_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(self.contract().model_dump_json(), encoding="utf-8")
            provider = JsonContractProvider(path)
            self.assertEqual(await provider.load("mock-v1"), self.contract())
            with self.assertRaisesRegex(LookupError, "unavailable"):
                await provider.load("other")

    async def test_json_contract_provider_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = JsonContractProvider(Path(directory) / "missing.json")
            with self.assertRaisesRegex(LookupError, "does not exist"):
                await missing.load("v1")

            invalid_path = Path(directory) / "invalid.json"
            invalid_path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(LookupError, "invalid"):
                await JsonContractProvider(invalid_path).load("v1")

    def test_builds_runtime_with_sigv4_gateway_and_opus(self) -> None:
        settings = AppSettings(
            environment=DeploymentEnvironment.TEST,
            aws_account_id="034362076319",
            aws_region="us-east-1",
            model_id="us.anthropic.claude-opus-4-6-v1",
            model_max_output_tokens=65_536,
            gateway_url="https://gateway.example.com/mcp",
            gateway_retrieve_tool_name="managed-kb___Retrieve",
            gateway_user_id="mock-e2e-principal",
            project_id="mock-prd-e2e",
            repository_name="agentcore-prd-agent",
            prd_contract_path=Path("/app/contracts/mock.json"),
            prompt_version="mock-v1",
            runtime_version="mock-v1",
        )
        with patch("prd_agent.application.SigV4MCPClientFactory") as factory:
            app = build_runtime_app(settings)
        self.assertIsInstance(app, BedrockAgentCoreApp)
        factory.assert_called_once_with(
            url="https://gateway.example.com/mcp",
            region_name="us-east-1",
        )
        paths = {getattr(route, "path", None) for route in app.routes}
        self.assertIn("/invocations", paths)
        self.assertIn("/ping", paths)


if __name__ == "__main__":
    unittest.main()
