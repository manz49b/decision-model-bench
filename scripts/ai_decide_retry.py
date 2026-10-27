"""Re-run ai_decide on test chats whose earlier ai_decide call errored, with the same tags, so each chat ends up with one
good answer. Scoring drops errored rows (report.load), so the earlier failures don't count twice.

    DATABRICKS_CONFIG_PROFILE=<your-profile> uv run python scripts/ai_decide_retry.py --dataset actions --questions v2
"""

import argparse

import pandas as pd
from databricks.connect import DatabricksSession
from dotenv import load_dotenv

from bench.data import dataset
from bench.engines.ai_decide import AiDecide
from bench.run import RESULTS, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["abcd", "actions"])
    ap.add_argument("--questions", required=True, choices=["v2", "v3"])
    a = ap.parse_args()
    load_dotenv()

    done = pd.read_csv(RESULTS, low_memory=False)
    done = done[(done["engine"] == "ai_decide") & (done["dataset"] == a.dataset) & (done["sample"] == "test300")
                & (done["question_set"] == a.questions)]
    answered = set(done.loc[done["error"].isna(), "case_id"].astype(str))
    data = dataset(a.dataset)
    if a.questions == "v3":
        from bench.prompting import Prompted

        data = Prompted(data, a.dataset)
    cases = data.sample("test300")
    todo = cases[~cases["id"].astype(str).isin(answered)]
    print(f"{len(todo)} of {len(cases)} chats still need an ai_decide answer")
    if todo.empty:
        return
    engine = AiDecide(DatabricksSession.builder.serverless(True).getOrCreate())
    tags = {"dataset": a.dataset, "sample": "test300", "question_set": a.questions}
    df = run([engine], todo, data.questions(a.questions), 1, tags)
    print(f"retried {df['case_id'].nunique()} chats, {df['error'].notna().sum()} errored rows")


if __name__ == "__main__":
    main()
