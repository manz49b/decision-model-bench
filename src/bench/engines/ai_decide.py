"""ai_decide: Databricks SQL function (Beta). Built for batches, so ``decide_table`` is the fair test."""

import json

from bench.engines import normalise


def sql_string(value: str) -> str:
    """A Spark SQL string literal. ai_decide needs ``questions`` as a constant, so it's inlined."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


class AiDecide:
    name = "ai_decide"

    def __init__(self, spark):
        self.spark = spark

    def decide(self, state: str, questions: dict) -> dict:
        sql = f"SELECT to_json(ai_decide(:state, {sql_string(json.dumps(questions))})) AS result"
        result = self.spark.sql(sql, args={"state": state}).collect()[0]["result"]
        meta = (json.loads(result) or {}).get("metadata") or {}
        self.last_meta = {"model": f"ai_decide-{meta.get('version', '?')}"}
        return self._normalise(result)

    def decide_table(self, table: str, questions: dict) -> list[dict]:
        """One query over every row of ``table`` (columns: id, state)."""
        sql = f"SELECT id, to_json(ai_decide(state, {sql_string(json.dumps(questions))})) AS result FROM {table}"
        rows = self.spark.sql(sql).collect()
        self.last_raw = {r["id"]: r["result"] for r in rows}
        return [{"id": r["id"], **self._normalise(r["result"])} for r in rows]

    @staticmethod
    def _normalise(result: str) -> dict:
        # Wrapped as {"error_message", "metadata", "response": {"answers": ...}}; yes/no answers use "probability"
        body = json.loads(result) or {}
        if body.get("error_message"):
            raise RuntimeError(f"ai_decide: {body['error_message']}")
        answers = (body.get("response") or {}).get("answers") or {}
        return normalise({q: {"noul": a["probability"]} if a.get("type") == "noul" else a for q, a in answers.items()})
