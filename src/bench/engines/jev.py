"""Jev: TypeSafe's hosted System One API."""

import os

import httpx

from bench.engines import meta, normalise

JEV_URL = "https://api.typesafe.ai/v1/systemone"


class Jev:
    name = "jev"

    def __init__(self, model: str = "jev-latest"):
        self.model = model
        self.client = httpx.Client(timeout=30, headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"})

    def decide(self, state: str, questions: dict) -> dict:
        """Retries server errors (e.g. 529 overloaded) with backoff, so load can't silently drop cases."""
        import time

        for attempt in range(5):
            response = self.client.post(JEV_URL, json={"model": self.model, "state": state, "questions": questions})
            if response.status_code < 500:
                break
            time.sleep(2 ** attempt)
        response.raise_for_status()
        body = response.json()
        self.last_meta = meta(body)
        return normalise(body["answers"])
