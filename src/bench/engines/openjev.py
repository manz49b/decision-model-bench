"""OpenJev on Databricks: ``system.ai.openjev-qwen35-4b`` (Qwen3.5 4B, hosted by Databricks) via Unity Gateway.

Not the same model as ``openjev/openjev`` on Hugging Face (27B, CC BY-NC).
Docs: https://docs.databricks.com/aws/en/machine-learning/model-serving/query-systemone-models
"""

import os

import httpx
from databricks.sdk import WorkspaceClient

from bench.engines import meta, normalise


class OpenJev:
    name = "openjev"

    def __init__(self, model: str | None = None):
        workspace = WorkspaceClient()
        self.model = model or os.environ.get("OPENJEV_MODEL", "system.ai.openjev-qwen35-4b")
        self.url = f"{workspace.config.host.rstrip('/')}/ai-gateway/typesafe/v1/systemone"
        self.client = httpx.Client(timeout=60, headers=workspace.config.authenticate())

    def decide(self, state: str, questions: dict) -> dict:
        response = self.client.post(self.url, json={"model": self.model, "state": state, "questions": questions})
        response.raise_for_status()
        body = response.json()
        self.last_meta = meta(body)
        return normalise(body["answers"])
