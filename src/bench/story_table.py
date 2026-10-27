"""Regenerate the results tables in RESULTS.md from results/runs.csv (story arms only: v2 criteria, test300).

    uv run python -m bench.story_table

Writes between ``<!-- results:start -->`` / ``<!-- results:end -->`` and ``<!-- detail:start -->`` /
``<!-- detail:end -->``. Missing runs show as "…" (pending) or "not run".
"""

import re
from pathlib import Path

import pandas as pd

from bench import metrics, report

STORY = Path(__file__).parents[2] / "RESULTS.md"
GPU_USD_PER_HOUR = 1.006  # AWS g5.xlarge (1× A10G 24 GB) on-demand, us-east-1, via Holori (Jul 2026)
SIZES = [100, 500, 2000]
# Optimised self-hosting is estimated, not measured (the laptop can't batch usefully):
# GPU seconds per call = 2 FLOPs per parameter per token × input tokens × forward passes / (peak × utilisation).
PARAMS = 2e9  # decider-2b
A10G_BF16_FLOPS = 125e12  # dense bf16 tensor peak (NVIDIA A10 datasheet; the A10G is the same GA102 part)
UTILISATION = 0.5  # typical for batched, prefill-only scoring with an optimised server

# (row label, engine name on job A, engine name on job B, where it runs, question set)
ROWS = [
    ("**Jev** (TypeSafe API)", "jev", "jev", "hosted", "v2"),
    ("**Jev, prompt-engineered** (+ 5 similar labelled chats)", "jev", "jev", "hosted", "v3"),
    ("**decider-2b** (open, no training)", "decider-2b", "decider-2b", "local", "v2"),
    *[(f"**decider-2b + adapter**, {n:,} chats" + (" (job B only)" if n == 2000 else ""),
       f"decider-2b+A{n}", f"decider-2b+B{n}", "local", "v2") for n in SIZES],
    ("**decider-2b, prompt-engineered** (no training)", "decider-2b", "decider-2b", "local", "v3"),
    ("**decider-2b + adapter**, 500 chats, **prompt-engineered**", "decider-2b+A500", "decider-2b+B500", "local", "v3"),
]
COLS = [("abcd", "flow"), ("abcd", "subflow"), ("abcd", "is_product_return"), ("actions", "next_action")]


def _story_df() -> dict[str, pd.DataFrame]:
    return {ds: report.load(ds) for ds in ("abcd", "actions")}


def _arm(df: pd.DataFrame, engine: str, qs: str) -> pd.DataFrame:
    return df[(df["engine"] == engine) & (df["question_set"] == qs)]


def thresholds() -> dict[str, float]:
    """Per engine, the yes/no threshold with the best F1 on job A's dev150, applied unchanged on test.

    A fixed 0.5 penalises calibrated models on a rare class (10% yes). Engines without dev scores keep 0.5.
    """
    dev = report.load("abcd", "dev150")
    dev = dev[dev["question_set"].isin(["v2", "v3"]) & (dev["question"] == "is_product_return")]
    out = {}
    for (engine, qs), g in dev.groupby(["engine", "question_set"]):
        gold, p = g["gold"].astype(str).eq("True").to_numpy(), g["score"].to_numpy(float)
        best = max(sorted(set(p)), key=lambda t: (_f1(gold, p >= t), -abs(t - 0.5)))
        out[engine, qs] = float(best)
    return out


def _f1(gold, pred) -> float:
    tp, fp, fn = (pred & gold).sum(), (pred & ~gold).sum(), (~pred & gold).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _with_threshold(g: pd.DataFrame, t: float) -> pd.DataFrame:
    """Rows with yes/no correctness recomputed at threshold ``t`` (for the paired test)."""
    noul = g["type"] == "noul"
    g = g.copy()
    g.loc[noul, "correct"] = (g.loc[noul, "score"] >= t) == g.loc[noul, "gold"].astype(str).eq("True")
    return g


def _value(g: pd.DataFrame, question: str, threshold: float = 0.5) -> dict | None:
    g = g[g["question"] == question]
    if g.empty:
        return None
    if g["type"].iat[0] == "noul":
        m = metrics.noul_summary(g.assign(gold=g["gold"].astype(str).eq("True")), threshold)
        return {"v": m["f1"], "lo": m["f1_lo"], "hi": m["f1_hi"], "ap": m["avg_precision"], "t": threshold}
    m = metrics.choice_summary(g)
    return {"v": m["accuracy"], "lo": m["acc_lo"], "hi": m["acc_hi"]}


def _speed(dfs, engine_a, engine_b, qs) -> dict[str, tuple[float, float]]:
    """Per job: (p50 ms per call, mean input tokens, questions per call). Kept per job: job A calls are ~4x longer than job B's, so a
    median over both mixed together depends on which jobs an arm ran."""
    out = {}
    for ds, engine in (("abcd", engine_a), ("actions", engine_b)):
        calls = _arm(dfs[ds], engine, qs).drop_duplicates(["run_id", "case_id", "repeat"])
        if not calls.empty:
            passes = _arm(dfs[ds], engine, qs).groupby(["run_id", "case_id", "repeat"])["question"].nunique().median()
            out[ds] = (calls["latency_ms"].median(), calls["input_tokens"].mean(), passes)
    return out


def _cost_one(where: str, p50_ms: float, tokens: float, passes: float = 1) -> float:
    """USD per 1,000 calls: Jev by token, self-hosted as estimated A10G time at full use (one pass per question)."""
    if where == "hosted":
        return tokens * 1000 / 1e6 * report.JEV_PRICE_PER_M_INPUT
    seconds = 2 * PARAMS * tokens * passes / (A10G_BF16_FLOPS * UTILISATION)
    return seconds / 3600 * GPU_USD_PER_HOUR * 1000


def _cost_cell(where: str, sp: tuple) -> str:
    """Self-hosted job A carries several questions: the low end shares the chat across them (prefix caching)."""
    if where == "hosted":
        return f"${_cost_one(where, *sp):.3f}"
    lo, hi = _cost_one(where, *sp[:2], 1), _cost_one(where, *sp)
    return f"~${hi:.3f}" if lo == hi else f"~${lo:.3f}–{hi:.3f}"


def _speed_cost(where: str, sp: dict) -> tuple[str, str]:
    """'A / B' p50 cells and cost per 1,000 calls (A / B), from per-job figures."""
    if not sp:
        return "…", "…"
    mark = "²" if where == "local" else ""
    speed = " / ".join(f"{sp[ds][0]:,.0f}" if ds in sp else "—" for ds in ("abcd", "actions")) + f" ms{mark}"
    cost = " / ".join(_cost_cell(where, sp[ds]) if ds in sp else "—" for ds in ("abcd", "actions"))
    return speed, cost + ("" if where == "hosted" else "¹")


def headline() -> str:
    dfs, th = _story_df(), thresholds()
    cells, best = {}, {c: 0.0 for c in COLS}
    for label, ea, eb, _, qs in ROWS:
        for ds, q in COLS:
            engine = ea if ds == "abcd" else eb
            v = _value(_arm(dfs[ds], engine, qs), q, th.get((engine, qs), 0.5))
            cells[label, ds, q] = v
            if v:
                best[ds, q] = max(best[ds, q], v["v"])
    lines = [
        "| Engine | A: `flow` (10) | A: `subflow` (54) | A: return yes/no (F1)³ | B: next action (30) | Speed, p50 per call (A / B) | Cost per 1,000 calls (A / B) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, ea, eb, where, qs in ROWS:
        row = [label]
        for c in COLS:
            v = cells[label, *c]
            row.append("…" if v is None else (f"**{v['v']:.2f}**" if abs(v["v"] - best[c]) < 1e-9 else f"{v['v']:.2f}"))
        speed, cost = _speed_cost(where, _speed(dfs, ea, eb, qs))
        row += [speed, cost]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_CACHE: dict = {}


def _test(dfs, ds, q, g, arm, ref, th) -> dict | None:
    """Paired exact McNemar of this arm against a reference arm on one question (first repeat of each).

    A pair tested in both directions is one test: the reverse direction reuses the same p (and Holm-adjusted p)."""
    if arm == ref:
        return None
    pair = (ds, q, frozenset([arm, ref]))
    if pair in _CACHE:
        t = _CACHE[pair]
        return {"p": t["p"], "better": not t["better"], "same_as": t} if t["arm"] != arm else t
    other = _with_threshold(_arm(dfs[ds], *ref), th.get(ref, 0.5))
    other, mine = other[other["question"] == q], g[g["question"] == q]
    if other.empty or mine.empty:
        return None
    t = metrics.mcnemar(mine, other)
    _CACHE[pair] = {"p": float(t["p_value"]), "better": t["only_a_right"] > t["only_b_right"], "arm": arm}
    return _CACHE[pair]


def holm(ps: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p values (family-wise error control, uniformly more powerful than Bonferroni)."""
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def _fmt(t: dict | None) -> str:
    if t is None:
        return "—"
    raw = "<0.001" if t["p"] < 0.001 else f"{t['p']:.3f}" if t["p"] < 0.05 else f"{t['p']:.2f}"
    if t["p"] >= 0.05:
        return f"{raw} (no difference)"
    verdict = "better" if t["better"] else "worse"
    return f"{raw} ({verdict}{', ✓ Holm' if t['holm'] < 0.05 else ', not after Holm'})"


def detail() -> str:
    """Every score with its 95% interval and paired tests, Holm-corrected across every test reported here."""
    dfs, th = _story_df(), thresholds()
    rows, tests = [], []
    _CACHE.clear()

    def add(t):
        if t is not None and "same_as" not in t and all(t is not x for x in tests):
            tests.append(t)
        return t

    for label, ea, eb, _, qs in ROWS:
        for ds, q in COLS:
            engine = ea if ds == "abcd" else eb
            t = th.get((engine, qs), 0.5)
            g = _with_threshold(_arm(dfs[ds], engine, qs), t)
            v = _value(g, q, t)
            if v is None:
                continue
            ts = [add(_test(dfs, ds, q, g, (engine, qs), ref, th)) for ref in (("jev", "v2"), ("jev", "v3"))]
            rows.append((label, ds, q, v, ts))
    grid = []
    for label, engine in [("base decider-2b", "decider-2b"), ("adapter A (triage)", "decider-2b+A500"),
                          ("adapter B (next action)", "decider-2b+B500")]:
        cells = []
        for ds, q in (("abcd", "subflow"), ("actions", "next_action")):
            g = _arm(dfs[ds], engine, "v2")
            v = _value(g, q)
            t = None if engine == "decider-2b" else add(_test(dfs, ds, q, g, (engine, "v2"), ("decider-2b", "v2"), {}))
            cells.append((v, t))
        grid.append((label, cells))
    # The comparisons the text relies on that aren't already a column: does training on examples beat the plain adapter?
    key = []
    for ds, q, a, b in [("abcd", "flow", "decider-2b+A500", "decider-2b+A500"), ("abcd", "subflow", "decider-2b+A500", "decider-2b+A500"),
                        ("abcd", "is_product_return", "decider-2b+A500", "decider-2b+A500"),
                        ("actions", "next_action", "decider-2b+B500", "decider-2b+B500")]:
        g = _with_threshold(_arm(dfs[ds], a, "v3"), th.get((a, "v3"), 0.5))
        key.append((ds, q, add(_test(dfs, ds, q, g, (a, "v3"), (b, "v2"), th))))
    for t, adj in zip(tests, holm([t["p"] for t in tests])):
        t["holm"] = adj
    for t in [c for r in rows for c in r[4]] + [t for *_, t in key] + [c[1] for _, cs in grid for c in cs]:
        if t is not None and "same_as" in t:
            t["holm"] = t["same_as"]["holm"]

    out = [(f"{len(tests)} paired tests in this section. ✓ Holm = still significant after Holm correction across all "
            f"{len(tests)} (family-wise 5%); \"not after Holm\" = p<0.05 on its own only, so treat as suggestive."), "",
           ("| Engine | Test | Score | 95% interval | vs Jev, criteria only (paired p) | vs Jev + examples (paired p) "
            "| Yes/no: threshold (from dev) · average precision |"),
           "|---|---|---|---|---|---|---|"]
    for label, ds, q, v, ts in rows:
        name = "F1" if q == "is_product_return" else "accuracy"
        ap = f"{v['t']:.2f} · {v['ap']:.2f}" if "ap" in v else ""
        out.append(f"| {label.replace('**', '')} | {'A' if ds == 'abcd' else 'B'}: `{q}` ({name}) | {v['v']:.3f} | "
                   f"{v['lo']:.2f}–{v['hi']:.2f} | {_fmt(ts[0])} | {_fmt(ts[1])} | {ap} |")
    out += ["", "**Does training on example-enriched prompts beat the plain 500-chat adapter?** (paired p)", "",
            "| Test | p |", "|---|---|"]
    out += [f"| {'A' if ds == 'abcd' else 'B'}: `{q}` | {_fmt(t)} |" for ds, q, t in key]
    out += ["", "**Right vs wrong adapter** (500 chats each; paired p vs the base model):", "",
            "| | A: `subflow` | B: next action |", "|---|---|---|"]
    for label, cells in grid:
        txt = [("…" if v is None else f"{v['v']:.2f}") + ("" if t is None else f", p {_fmt(t)}") for v, t in cells]
        out.append(f"| {label} | {txt[0]} | {txt[1]} |")
    return "\n".join(out)


def write() -> None:
    s = STORY.read_text()
    for tag, body in (("results", headline()), ("detail", detail())):
        s = re.sub(rf"<!-- {tag}:start -->.*?<!-- {tag}:end -->", f"<!-- {tag}:start -->\n{body}\n<!-- {tag}:end -->",
                   s, flags=re.DOTALL)
    STORY.write_text(s)


if __name__ == "__main__":
    write()
    print(headline())
