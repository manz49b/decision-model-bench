#!/usr/bin/env bash
# What's left of the story after the restart: B2000, dev scores for yes/no thresholds, then A2000 (optional, last).
set -euo pipefail
cd "$(dirname "$0")/.."
export CC="$PWD/.bin/cc"
quiet() { grep --line-buffered -v -i "warn\|causal_conv1d\|Loading weights"; }
run() {  # model-path label dataset sample
  uv run python -m bench.run --engine decider --path "$1" --label "$2" --temps "results/temps_${2}_${3}.json" \
    --dataset "$3" --sample "$4" --questions v2 2>&1 | quiet | tail -1
}
calibrate() { uv run python -m bench.lora.calibrate "$1" "results/temps_${2}_${3}.json" v2 "$3" 2>&1 | grep "\[calibrated\]"; }
train() {  # job dataset n
  uv run python -m bench.lora.train --dataset "$2" --questions v2 --train "$3" --eval 100 --out "$1$3" 2>&1 \
    | grep --line-buffered -E "\[eval|\[sched|\[train\] step [0-9]*0/|Traceback|Error" | tee "results/train_$1$3.log"
}
merge() { uv run python -m bench.lora.merge "adapters/$1" 2>&1 | tail -1; }
stage() { echo "[stage] $*"; uv run python -m bench.story_table >/dev/null 2>&1 || true; }

stage "B2000"
train B actions 2000; merge B2000
calibrate adapters/B2000-merged decider-2b+B2000 actions
run adapters/B2000-merged decider-2b+B2000 actions test300
rm -rf adapters/B2000-merged

stage "dev scores for yes/no thresholds (job A)"
run Mapika/decider-2b decider-2b abcd dev150
for n in 100 500; do
  merge "A$n"; run "adapters/A$n-merged" "decider-2b+A$n" abcd dev150; rm -rf "adapters/A$n-merged"
done

stage "A2000 (optional: safe to stop here)"
train A abcd 2000; merge A2000
calibrate adapters/A2000-merged decider-2b+A2000 abcd
run adapters/A2000-merged decider-2b+A2000 abcd test300
run adapters/A2000-merged decider-2b+A2000 abcd dev150
rm -rf adapters/A2000-merged
stage "done"
echo "[story done]"
