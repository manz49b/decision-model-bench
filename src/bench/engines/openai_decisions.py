"""OpenAI Decisions API (gpt-6-luna). Same questions as Jev, translated to OpenAI's format: ``state`` becomes ``input``,
``noul`` becomes ``predicate``, and ``criteria`` maps become ``choices`` lists.

Docs: https://developers.openai.com/api/docs/guides/decisions
"""

import os
import time

import httpx

from bench.engines import meta

URL = "https://api.openai.com/v1/decisions"


def to_openai(questions: dict) -> list[dict]:
    out = []
    for name, spec in questions.items():
        q = {"name": name, "instructions": spec["instructions"]}
        if spec["type"] == "noul":
            q["type"] = "predicate"
        else:
            q["type"] = "choice"
            q["choices"] = [{"value": k, "description": d or k} for k, d in spec["criteria"].items()]
        out.append(q)
    return out


class OpenAIDecisions:
    name = "openai_decisions"

    def __init__(self, model: str = "gpt-6-luna"):
        self.model = model
        self.client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})

    def decide(self, state: str, questions: dict) -> dict:
        """Retries rate limits and server errors with backoff, so load can't silently drop cases."""
        body = {"model": self.model, "input": state, "questions": to_openai(questions)}
        for attempt in range(6):
            response = self.client.post(URL, json=body)
            if response.status_code != 429 and response.status_code < 500:
                break
            if "insufficient_quota" in response.text:  # out of credit: retrying won't help
                break
            time.sleep(2 ** attempt)
        response.raise_for_status()
        data = response.json()
        self.last_meta = meta(data)
        return self._normalise(data["answers"])

    @staticmethod
    def _normalise(answers: list) -> dict:
        """To the shared shape: choice -> (label, confidence); predicate -> (None, probability of yes). Refusals are
        kept as an empty answer, so they count as wrong rather than crashing the run."""
        out = {}
        for a in answers:
            if a["type"] == "choice":
                out[a["name"]] = {"label": a["choice"], "score": a.get("confidence")}
            elif a["type"] == "predicate":
                out[a["name"]] = {"label": None, "score": a["probability"]}
            else:
                out[a["name"]] = {"label": None, "score": None}
        return out
