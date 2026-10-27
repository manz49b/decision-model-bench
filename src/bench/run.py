"""Run cases through engines and append one row per case × question × repeat to results/runs.csv.

Controls for a fair measurement:
- every engine gets the identical ``state`` and ``questions`` (same frozen sample, same question set);
- a warm-up call per engine first, not logged, so connection setup doesn't land in the latency numbers;
- ``repeats`` full passes, each in a fresh random case order, with engines interleaved per case,
  so time-of-day or rate-limit drift spreads over every engine instead of biasing one;
- the model version the engine reports is logged, since ``jev-latest`` can change under us.
"""

import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).parents[2] / "results" / "runs.csv"


def shuffled(questions: dict, rng: random.Random) -> dict:
    """The same questions with every ``criteria`` map in a random order: a check for option-position bias."""
    out = {}
    for name, spec in questions.items():
        if "criteria" in spec:
            keys = list(spec["criteria"])
            rng.shuffle(keys)
            spec = {**spec, "criteria": {k: spec["criteria"][k] for k in keys}}
        out[name] = spec
    return out


def run(engines: list, cases: pd.DataFrame, questions: dict, repeats: int = 1, tags: dict | None = None,
        seed: int = 0, shuffle_criteria: bool = False) -> pd.DataFrame:
    """``cases`` has columns id, state, and one gold column per question name."""
    run_id, rows, rng = uuid.uuid4().hex[:8], [], random.Random(seed)
    for engine in engines:  # warm-up, not logged
        engine.decide(cases["state"].iat[0], questions)
    records = cases.to_dict("records")
    for repeat in range(repeats):
        rng.shuffle(records)
        for case in records:
            for engine in rng.sample(engines, k=len(engines)):
                start = time.perf_counter()
                asked = shuffled(questions, rng) if shuffle_criteria else questions
                try:
                    answers, error = engine.decide(case["state"], asked), None
                except Exception as e:  # noqa: BLE001  (log the failure and keep going)
                    answers, error = {}, f"{type(e).__name__}: {e}"[:300]
                ms = (time.perf_counter() - start) * 1000
                call_meta = getattr(engine, "last_meta", {}) if not error else {}
                for q, spec in questions.items():
                    a = answers.get(q, {})
                    rows.append({
                        "run_id": run_id, "at": datetime.now(UTC), "engine": engine.name, **(tags or {}),
                        "model": call_meta.get("model"), "input_tokens": call_meta.get("input_tokens"),
                        "repeat": repeat, "case_id": case["id"], "question": q, "type": spec["type"],
                        "gold": case[q], "label": a.get("label"), "score": a.get("score"), "latency_ms": ms,
                        "error": error,
                    })
    df = pd.DataFrame(rows)
    append(df)
    return df


def append(df: pd.DataFrame) -> None:
    """Append rows under the file's existing header. A new column rewrites the file with it, so rows never shift.

    Holds an exclusive lock, so parallel runs can't interleave their rows."""
    import fcntl

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _append(df)


def _append(df: pd.DataFrame) -> None:
    if not RESULTS.exists():
        df.to_csv(RESULTS, index=False)
        return
    header = pd.read_csv(RESULTS, nrows=0).columns.tolist()
    if set(df.columns) - set(header):
        pd.concat([pd.read_csv(RESULTS, low_memory=False), df]).to_csv(RESULTS, index=False)
    else:
        df.reindex(columns=header).to_csv(RESULTS, mode="a", header=False, index=False)


def main():
    import argparse

    from dotenv import load_dotenv

    from bench.data import dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="jev", choices=["jev", "openjev", "decider", "ai_decide", "openai_decisions"])
    ap.add_argument("--dataset", default="abcd", choices=["abcd", "actions"], help="abcd = job A triage, actions = job B")
    ap.add_argument("--sample", default="test300")
    ap.add_argument("--questions", default="v1", help="question set version, see abcd.questions()")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--label", default=None, help="engine name in the results, e.g. decider-2b+lora500")
    ap.add_argument("--no-examples", action="store_true", help="v3 ablation: no retrieved examples")
    ap.add_argument("--playbook", action="store_true", help="v3 ablation: add the playbook (job B; hurt on dev)")
    ap.add_argument("--twostep", action="store_true", help="v3 ablation: two-step subflow (job A; hurt on dev)")
    ap.add_argument("--shuffle-criteria", action="store_true", help="random criteria order per call")
    ap.add_argument("--max-turns", type=int, default=None, help="only the first N turns of each chat (early triage)")
    ap.add_argument("--temps", default=None, help="decider: JSON file of per-type temperatures (bench.lora.calibrate)")
    ap.add_argument("--path", default="Mapika/decider-2b", help="decider: model id or merged-adapter folder")
    a = ap.parse_args()
    load_dotenv()
    if a.engine == "jev":
        from bench.engines.jev import Jev as E
    elif a.engine == "openjev":
        from bench.engines.openjev import OpenJev as E
    else:
        from bench.engines.decider import Decider

    if a.engine == "openai_decisions":
        from bench.engines.openai_decisions import OpenAIDecisions

        engine = OpenAIDecisions()
    elif a.engine == "ai_decide":
        from databricks.connect import DatabricksSession

        from bench.engines.ai_decide import AiDecide

        engine = AiDecide(DatabricksSession.builder.serverless(True).getOrCreate())
    elif a.engine == "decider":
        import json

        temps = json.loads(Path(a.temps).read_text()) if a.temps else None
        engine = Decider(a.path, a.label or "decider-2b", temps)
    else:
        engine = E()
    engine.name = a.label or engine.name
    data = dataset(a.dataset)
    if a.questions == "v3":
        from bench.prompting import Prompted, TwoStep

        data = Prompted(data, a.dataset, examples=not a.no_examples, use_playbook=a.playbook)
        if a.dataset == "abcd" and a.twostep:
            engine = TwoStep(engine)
    ablation = "".join(x for x, on in [("-noexamples", a.no_examples), ("+playbook", a.playbook),
                                       ("+twostep", a.twostep)] if on)
    variant = "+".join(v for v in [a.questions + ablation, "shuffled" if a.shuffle_criteria else "",
                                   f"first{a.max_turns}turns" if a.max_turns else ""] if v)
    cases = data.sample(a.sample, a.max_turns) if a.max_turns else data.sample(a.sample)
    tags = {"dataset": a.dataset, "sample": a.sample, "question_set": variant}
    df = run([engine], cases, data.questions(a.questions), a.repeats, tags,
             shuffle_criteria=a.shuffle_criteria)
    print(f"run_id {df['run_id'].iat[0]}: {df['case_id'].nunique()} cases × {a.repeats} repeats, "
          f"{df['error'].notna().sum()} errored rows, model {df['model'].dropna().unique()}")


if __name__ == "__main__":
    main()
