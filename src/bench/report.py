"""Summarise results/runs.csv: accuracy with intervals per engine × question set × question, latency, cost."""

import pandas as pd

from bench import metrics
from bench.run import RESULTS

JEV_PRICE_PER_M_INPUT = 0.042  # USD, TypeSafe pricing page; output tokens are free
KEYS = ["engine", "question_set"]


def load(dataset: str = "abcd", sample: str = "test300") -> pd.DataFrame:
    df = pd.read_csv(RESULTS, low_memory=False)
    df["dataset"] = df["dataset"].fillna("abcd") if "dataset" in df else "abcd"  # rows before job B are all job A
    df = df[(df["dataset"] == dataset) & (df["sample"] == sample) & df["error"].isna()].copy()
    df["correct"] = df["gold"].astype(str) == df["label"].astype(str)
    noul = df["type"] == "noul"
    df.loc[noul, "correct"] = (df.loc[noul, "score"] >= 0.5) == df.loc[noul, "gold"].astype(str).eq("True")
    df["arm"] = df["engine"] + " · " + df["question_set"]
    return df


def paired(df: pd.DataFrame, baseline: str = "jev · v1") -> pd.DataFrame:
    """McNemar exact test of every arm against ``baseline``, per question, on the cases both answered."""
    rows = []
    for question, g in df.groupby("question"):
        base = g[g["arm"] == baseline]
        for arm, other in g[g["arm"] != baseline].groupby("arm"):
            t = metrics.mcnemar(other, base)
            rows.append({"question": question, "arm": arm, "vs": baseline, **t,
                         "acc_diff": other[other["repeat"] == 0]["correct"].mean() - base[base["repeat"] == 0]["correct"].mean()})
    return pd.DataFrame(rows).round(4)


def accuracy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    choice, noul = [], []
    for (*key, question, qtype), g in df.groupby([*KEYS, "question", "type"]):
        row = dict(zip(KEYS, key), question=question)
        if qtype == "noul":
            noul.append({**row, **metrics.noul_summary(g.assign(gold=g["gold"].astype(str).eq("True")))})
        else:
            choice.append({**row, **metrics.choice_summary(g)})
    return pd.DataFrame(choice).round(3), pd.DataFrame(noul).round(3)


def latency_and_cost(df: pd.DataFrame) -> pd.DataFrame:
    calls = df.drop_duplicates(["run_id", "engine", "case_id", "repeat"])
    out = calls.groupby(KEYS).agg(
        calls=("latency_ms", "size"),
        p50_ms=("latency_ms", "median"),
        p95_ms=("latency_ms", lambda s: s.quantile(0.95)),
        max_ms=("latency_ms", "max"),
        mean_input_tokens=("input_tokens", "mean"),
        models=("model", lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))),
    )
    out["usd_per_1k_calls"] = out["mean_input_tokens"] * 1000 / 1e6 * JEV_PRICE_PER_M_INPUT
    out.loc[~out.index.get_level_values("engine").str.startswith("jev"), "usd_per_1k_calls"] = None
    return out.round(4)


if __name__ == "__main__":
    import sys

    df = load(*(sys.argv[1:2] or ["abcd"]))
    choice, noul = accuracy(df)
    print(choice.to_string(index=False), "\n")
    print(noul.to_string(index=False), "\n")
    print(latency_and_cost(df).to_string(), "\n")
    print(metrics.consistency(df).round(3).to_string(), "\n")
    print(paired(df).to_string(index=False))
