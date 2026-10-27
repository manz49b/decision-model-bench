"""Wait for the ai_decide batch run's billing rows (system.billing.usage lags hours), then print its DBU and list cost.

    DATABRICKS_CONFIG_PROFILE=<your-profile> uv run python scripts/ai_decide_batch_bill.py
"""

import json
import time
from pathlib import Path

from databricks.connect import DatabricksSession

RUN = json.loads((Path(__file__).parents[1] / "results" / "ai_decide_batch.json").read_text())
WINDOW_END = "2026-10-27 08:20"  # billing comes in 10-minute buckets; my next serverless use was the 08:26 check


def main():
    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    me = spark.sql("SELECT current_user() u").collect()[0].u
    start = RUN["start"][:16].replace("T", " ")
    for _ in range(24):  # up to 8 hours
        latest = spark.sql("SELECT max(usage_end_time) t FROM system.billing.usage").collect()[0].t
        if str(latest) >= "2026-10-27 08:30":
            break
        print(f"billing has data up to {latest}, waiting", flush=True)
        time.sleep(1200)
    rows = spark.sql(f"""
        SELECT u.billing_origin_product product, u.usage_start_time start, u.usage_quantity dbu,
               u.usage_quantity * p.pricing.effective_list.default usd
        FROM system.billing.usage u
        LEFT JOIN system.billing.list_prices p ON u.sku_name = p.sku_name AND u.cloud = p.cloud
          AND u.usage_start_time >= p.price_start_time AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
        WHERE u.usage_start_time >= '{start[:15]}0' AND u.usage_start_time < '{WINDOW_END}'
          AND u.identity_metadata.run_as = '{me}'""").collect()
    for r in rows:
        print(r.product, r.start, round(r.dbu, 4), "DBU", round(r.usd or 0, 4), "USD")
    dbu, usd = sum(r.dbu for r in rows), sum(r.usd or 0 for r in rows)
    print(f"TOTAL {dbu:.4f} DBU, ${usd:.4f} at list, for {RUN['chats']} chats = ${usd / RUN['chats'] * 1000:.3f} per 1,000")


if __name__ == "__main__":
    main()
