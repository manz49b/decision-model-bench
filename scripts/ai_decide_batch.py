"""ai_decide as one batch query over the 300 next-action test chats, with the same state and questions as the
single-call run, so the billed compute can be compared like for like. Prints the start and end time to find the
billing rows (system.billing.usage lags by a few hours).

    DATABRICKS_CONFIG_PROFILE=<your-profile> uv run python scripts/ai_decide_batch.py
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from databricks.connect import DatabricksSession

from bench.data import dataset
from bench.engines.ai_decide import AiDecide
from bench.run import RESULTS

OUT = Path(__file__).parents[1] / "results" / "ai_decide_batch.json"


def main():
    data = dataset("actions")
    questions = data.questions("v2")
    cases = data.sample("test300")[["id", "state"]]
    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    spark.createDataFrame(cases).createOrReplaceTempView("ai_decide_batch_cases")

    start = datetime.now(UTC)
    t0 = time.perf_counter()
    rows = AiDecide(spark).decide_table("ai_decide_batch_cases", questions)
    seconds = time.perf_counter() - t0
    end = datetime.now(UTC)

    # Same input as the single calls: the answers should match the single-call run on (nearly) every chat
    single = pd.read_csv(RESULTS, low_memory=False)
    single = single[(single["engine"] == "ai_decide") & (single["dataset"] == "actions")
                    & (single["question_set"] == "v2") & single["error"].isna()]
    single = dict(zip(single["case_id"].astype(str), single["label"]))
    batch = {str(r["id"]): r["next_action"]["label"] for r in rows}
    same = sum(batch[c] == single.get(c) for c in batch) / len(batch)

    print(f"{len(rows)} chats in {seconds:.0f}s, from {start:%Y-%m-%d %H:%M:%S} to {end:%H:%M:%S} UTC")
    print(f"same answer as the single-call run: {same:.1%}")
    OUT.write_text(json.dumps({"start": start.isoformat(), "end": end.isoformat(), "seconds": seconds,
                               "chats": len(rows), "same_as_single": same}, indent=1))


if __name__ == "__main__":
    main()
