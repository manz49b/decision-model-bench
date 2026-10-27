#!/usr/bin/env bash
# OpenAI Decisions on the same test chats and questions as every other engine: both jobs, criteria only and + examples
set -uo pipefail
cd "$(dirname "$0")/.."
for ds in abcd actions; do
  for q in v2 v3; do
    echo "== $ds $q $(date +%T)"
    uv run python -m bench.run --engine openai_decisions --dataset "$ds" --sample test300 --questions "$q" 2>&1 | tail -2
  done
done
echo "== done $(date +%T)"
