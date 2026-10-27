"""Metrics with uncertainty, over results rows (one per engine × case × question × repeat).

- Every headline number has a 95% bootstrap interval, resampling *cases* (not rows), since a case's three
  questions and its repeats aren't independent.
- ``noul`` questions are imbalanced (10% yes), so accuracy is reported next to the majority baseline, with
  precision, recall, F1, average precision (threshold-free) and calibration (Brier, ECE).
- Engine-vs-engine gaps use a paired test (McNemar), since every engine answers the same cases.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score

N_BOOT = 2000


def _bootstrap(df: pd.DataFrame, counts: dict, stat, seed: int = 0) -> tuple[float, float, float]:
    """``stat`` on summed per-case ``counts``, and a 95% interval from resampling cases with replacement.

    ``counts`` maps a name to a row-level 0/1 column; each case's rows (questions, repeats) are summed first, so a
    resample draws whole cases. Vectorised: one matrix product instead of rebuilding a DataFrame per draw.
    """
    per_case = df.assign(**counts).groupby("case_id")[list(counts)].sum().to_numpy(float)  # [cases, k]
    rng = np.random.default_rng(seed)
    n = len(per_case)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(float)  # [boots, cases]
    boots = stat(weights @ per_case)  # [boots, k] -> [boots]
    point = stat(per_case.sum(0, keepdims=True))[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def _accuracy(sums: np.ndarray) -> np.ndarray:
    return sums[:, 0] / sums[:, 1]


def _f1(sums: np.ndarray) -> np.ndarray:
    tp, fp, fn = sums[:, 0], sums[:, 1], sums[:, 2]
    return np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)


def ece(gold: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: how far predicted probabilities are from observed frequencies."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(gold[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def choice_summary(df: pd.DataFrame) -> dict:
    acc = _bootstrap(df, {"hit": (df["gold"] == df["label"]).astype(int), "n": 1}, _accuracy)
    majority = df["gold"].value_counts(normalize=True).iat[0]
    return {
        "n_cases": df["case_id"].nunique(),
        "accuracy": acc[0], "acc_lo": acc[1], "acc_hi": acc[2],
        "macro_f1": f1_score(df["gold"], df["label"].fillna(""), average="macro"),
        "majority_baseline": majority,
    }


def noul_summary(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    gold, p = df["gold"].astype(bool).to_numpy(), df["score"].to_numpy(float)
    pred = p >= threshold
    tp, fp, fn = (pred & gold).sum(), (pred & ~gold).sum(), (~pred & gold).sum()
    f1 = _bootstrap(df, {"tp": (pred & gold).astype(int), "fp": (pred & ~gold).astype(int),
                         "fn": (~pred & gold).astype(int)}, _f1)
    return {
        "n_cases": df["case_id"].nunique(),
        "accuracy": (pred == gold).mean(),
        "majority_baseline": max(gold.mean(), 1 - gold.mean()),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "f1": f1[0], "f1_lo": f1[1], "f1_hi": f1[2],
        "avg_precision": average_precision_score(gold, p),
        "brier": ((p - gold) ** 2).mean(), "ece": ece(gold, p),
    }


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Across repeats of the same case: share of cases whose answer never changed, and mean score spread."""
    def per(g):
        answer = g["label"] if g["type"].iat[0] == "choice" else g["score"] >= 0.5
        return pd.Series({"same_answer": answer.nunique() == 1, "score_range": g["score"].max() - g["score"].min()})

    keys = ["arm"] if "arm" in df else ["engine"]  # an arm is engine + question set: never mix question sets
    per_case = df.groupby([*keys, "question", "case_id"]).apply(per, include_groups=False)
    return per_case.groupby([*keys, "question"]).agg(
        cases=("same_answer", "size"), stable_share=("same_answer", "mean"), mean_score_range=("score_range", "mean")
    )


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired exact McNemar test on per-case correctness (repeat 0 of each), for one question."""
    ca = a[a["repeat"] == 0].set_index("case_id")["correct"]
    cb = b[b["repeat"] == 0].set_index("case_id")["correct"]
    ca, cb = ca.align(cb, join="inner")
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return {"cases": len(ca), "only_a_right": only_a, "only_b_right": only_b, "p_value": p}
