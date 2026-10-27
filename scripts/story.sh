#!/usr/bin/env bash
# The story's GPU work. Every decider arm, base included: temperatures refitted on that job's dev150, then scored on
# that job's test300, all with the company criteria (v2). Adapters: A<n> = job A (triage), B<n> = job B (next action).
set -euo pipefail
cd "$(dirname "$0")/.."
export CC="$PWD/.bin/cc"
quiet() { grep --line-buffered -v -i "warn\|causal_conv1d\|Loading weights"; }

score() {  # model-path label dataset
  local temps="results/temps_${2}_${3}.json"
  uv run python -m bench.lora.calibrate "$1" "$temps" v2 "$3" 2>&1 | grep "\[calibrated\]"
  uv run python -m bench.run --engine decider --path "$1" --label "$2" --temps "$temps" \
    --dataset "$3" --sample test300 --questions v2 2>&1 | quiet | tail -1
}
train() {  # job dataset n
  uv run python -m bench.lora.train --dataset "$2" --questions v2 --train "$3" --eval 100 --out "$1$3" 2>&1 \
    | grep --line-buffered -E "\[eval|\[sched|\[train\] step [0-9]*0/|Traceback|Error" | tee "results/train_$1$3.log"
  uv run python -m bench.lora.merge "adapters/$1$3" 2>&1 | tail -1
}
arm() {  # job dataset n [cross-dataset]: train, merge, score on its own job (and the other job), clean up
  train "$1" "$2" "$3"
  score "adapters/$1$3-merged" "decider-2b+$1$3" "$2"
  [ -n "${4:-}" ] && score "adapters/$1$3-merged" "decider-2b+$1$3" "$4"
  rm -rf "adapters/$1$3-merged"
}

echo "[stage] base"
score Mapika/decider-2b decider-2b abcd
score Mapika/decider-2b decider-2b actions
echo "[stage] job A adapters"
arm A abcd 100
arm A abcd 500 actions      # also scored on job B: the wrong-adapter test
echo "[stage] job B adapters"
arm B actions 100
arm B actions 500 abcd      # also scored on job A: the wrong-adapter test
arm B actions 2000
echo "[stage] job A 2000"
arm A abcd 2000
uv run python -m bench.report abcd > results/report_abcd.txt 2>&1
uv run python -m bench.report actions > results/report_actions.txt 2>&1
echo "[story done]"
