"""Refit decider's per-type temperatures on dev150, as decider's own recipe does after every training stage.

Why: decider serves ``noul`` at temperature 0.05, i.e. logits ×20. LoRA is trained at temperature 1, so a small
shift in the yes/no logits becomes a huge one when served, and rounding to 4 decimals then collapses most scores
to exactly 0 or 1 (ties that also wreck the ranking). Applied to every decider arm, base included, so all
of them get the same calibration step. Fitted on dev only; test is never seen.

    uv run python -m bench.lora.calibrate Mapika/decider-2b results/temps_decider-2b.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from decider.infer import Decider

from bench.data import dataset

GRID = np.logspace(-3, 1, 400)  # 0.001–10: A500s2 hit the old 0.01 floor
UNIT = {"choice": 1.0, "noul": 1.0, "score": 1.0}


def collect(path: str, questions: dict, data) -> dict[str, list[tuple[np.ndarray, int]]]:
    """(probabilities at temperature 1, gold index) per answer type, over dev150."""
    model = Decider(path, use_graphs=False, temperature_by_type=UNIT)
    rows = {"choice": [], "noul": []}
    for _, case in data.sample("dev150").iterrows():
        answers = model.system_one(case["state"], questions)["answers"]
        for name, spec in questions.items():
            a = answers[name]
            if spec["type"] == "noul":
                rows["noul"].append((np.array([1 - a["noul"], a["noul"]]), int(bool(case[name]))))
            else:
                names = list(spec["criteria"])
                rows["choice"].append((np.array([a["probabilities"][n] for n in names]), names.index(case[name])))
    return rows


def fit(rows: list[tuple[np.ndarray, int]]) -> float:
    """The temperature minimising NLL of the gold option: p_T ∝ p^(1/T)."""
    def nll(t):
        total = 0.0
        for p, gold in rows:
            logp = np.log(np.clip(p, 1e-6, 1)) / t
            total -= logp[gold] - np.logaddexp.reduce(logp)
        return total

    return float(GRID[np.argmin([nll(t) for t in GRID])])


def main(path: str, out: str, questions: str = "v1", name: str = "abcd") -> dict:
    data = dataset(name)
    if questions == "v3":
        from bench.prompting import Prompted

        data = Prompted(data, name)
    rows = collect(path, data.questions(questions), data)
    # Probabilities came from T=1, so the fitted value is the temperature relative to T=1: use it directly
    temps = {t: fit(rows[t]) if rows[t] else 1.0 for t in ("choice", "noul")} | {"score": 1.0}
    Path(out).write_text(json.dumps(temps, indent=1))
    print("[calibrated]", path, temps)
    return temps


if __name__ == "__main__":
    main(*sys.argv[1:5])
