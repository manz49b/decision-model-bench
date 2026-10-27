#!/usr/bin/env bash
# ai_decide on the same test chats and questions as every other engine: both jobs, criteria only (v2) and + examples (v3)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${DATABRICKS_CONFIG_PROFILE:?set DATABRICKS_CONFIG_PROFILE to your Databricks CLI profile}"
for ds in abcd actions; do
  for q in v2 v3; do
    echo "== $ds $q $(date +%T)"
    uv run python -m bench.run --engine ai_decide --dataset "$ds" --sample test300 --questions "$q" 2>&1 \
      | grep -v -i -E "warn|skills are not" | tail -3
  done
done
echo "== done $(date +%T)"
