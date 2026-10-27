"""Which model does ai_decide run? Send the same chats and questions to ai_decide, the Jev API and Databricks' OpenJev
endpoint, then compare answers and scores pairwise. The same model should agree on nearly every answer, with near-equal
scores; a different model can score similarly overall and still disagree chat by chat.

    DATABRICKS_CONFIG_PROFILE=<your-profile> uv run python scripts/ai_decide_check.py --n 150
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from databricks.connect import DatabricksSession
from dotenv import load_dotenv

from bench.data import dataset
from bench.engines.ai_decide import AiDecide
from bench.engines.jev import Jev
from bench.engines.openjev import OpenJev

OUT = Path(__file__).parents[1] / "results" / "ai_decide_check.json"


def main():
    load_dotenv()  # JEV_API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="dev chats (job A, v2 questions)")
    a = ap.parse_args()
    data = dataset("abcd")
    questions = data.questions("v2")
    cases = data.sample("dev150").head(a.n)[["id", "state"]]

    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    spark.createDataFrame(cases).createOrReplaceTempView("ai_decide_check_cases")
    ai = AiDecide(spark)
    answers = {"ai_decide": {r.pop("id"): r for r in ai.decide_table("ai_decide_check_cases", questions)}}
    print("ai_decide raw output, first chat:", next(iter(ai.last_raw.values())))
    for q in questions:
        missing = sum(1 for a in answers["ai_decide"].values() if a.get(q, {}).get("label") is None and a.get(q, {}).get("score") is None)
        print(f"ai_decide unanswered on {q}: {missing}/{len(cases)}")
    # Pin both Jev versions: jev-latest is moving from 1.13.0 to 1.14.0, and ai_decide could be on either
    for name, engine in (("jev-1.13.0", Jev("jev-1.13.0")), ("jev-1.14.0", Jev("jev-1.14.0")), ("openjev", OpenJev())):
        answers[name] = {c.id: engine.decide(c.state, questions) for c in cases.itertuples()}
        print(name, "answered by", getattr(engine, "last_meta", {}).get("model"))

    rows = []
    for x, y in combinations(answers, 2):
        for q in questions:
            same, gaps = [], []
            for cid in cases["id"]:
                ax, ay = answers[x][cid].get(q, {}), answers[y][cid].get(q, {})
                if questions[q]["type"] == "noul":  # yes/no has no label: compare which side of 0.5 each lands
                    sx, sy = ax.get("score"), ay.get("score")
                    same.append(sx is not None and sy is not None and (sx >= 0.5) == (sy >= 0.5))
                else:
                    same.append(ax.get("label") is not None and ax.get("label") == ay.get("label"))
                if ax.get("score") is not None and ay.get("score") is not None:
                    gaps.append(abs(ax["score"] - ay["score"]))
            rows.append({"pair": f"{x} vs {y}", "question": q, "same_answer": round(sum(same) / len(same), 3),
                         "mean_score_gap": round(sum(gaps) / len(gaps), 4) if gaps else None})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    OUT.write_text(json.dumps({"n": len(cases), "pairs": rows, "answers": answers, "ai_decide_raw": ai.last_raw},
                              indent=1, default=str))
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
