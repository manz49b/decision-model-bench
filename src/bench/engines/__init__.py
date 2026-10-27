"""Every engine takes the same ``state`` and ``questions`` and returns answers in one normalised shape.

``{question_name: {"label": <criteria key or None>, "score": <0.0 to 1.0>}}``: for ``choice`` questions
``score`` is the probability of the chosen label, for ``noul`` questions it's the probability of yes.

After each call ``engine.last_meta`` holds what the engine reported about it: the model version that actually
answered (``jev-latest`` resolves to a pinned version) and token usage, for cost.
"""

from typing import Protocol


class Engine(Protocol):
    name: str
    last_meta: dict

    def decide(self, state: str, questions: dict) -> dict: ...


def normalise(answers: dict) -> dict:
    """System One answers (Jev, OpenJev, decider all share the wire format) to the normalised shape."""
    out = {}
    for question, answer in answers.items():
        if "choice" in answer:
            out[question] = {"label": answer["choice"], "score": answer.get("confidence")}
        else:
            out[question] = {"label": None, "score": answer.get("noul")}
    return out


def meta(body: dict) -> dict:
    usage = body.get("usage") or {}
    return {"model": body.get("model"), "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
