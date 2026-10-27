#!/usr/bin/env bash
# Prompt-engineered (v3) decider arms: base, then 500-chat adapters trained on v3 inputs (same 500 chats as the
# example pool, leave-one-out). Each: temperatures refitted on dev, scored on test (and dev for job A's threshold).
set -euo pipefail
cd "$(dirname "$0")/.."
export CC="$PWD/.bin/cc"
quiet() { grep --line-buffered -v -i "warn\|causal_conv1d\|Loading weights"; }
temps() { echo "results/temps_${1}_${2}_v3.json"; }
calibrate() { uv run python -m bench.lora.calibrate "$1" "$(temps "$2" "$3")" v3 "$3" 2>&1 | grep "\[calibrated\]"; }
run() {  # path label dataset sample
  uv run python -m bench.run --engine decider --path "$1" --label "$2" --temps "$(temps "$2" "$3")" \
    --dataset "$3" --sample "$4" --questions v3 2>&1 | quiet | tail -1
}
arm() {  # path label dataset
  calibrate "$1" "$2" "$3"; run "$1" "$2" "$3" test300
  if [ "$3" = abcd ]; then run "$1" "$2" "$3" dev150; fi
  uv run python -m bench.story_table >/dev/null 2>&1 || true
}
adapter() {  # job dataset
  echo "[stage] ${1}500 trained on prompt-engineered inputs"
  uv run python -m bench.lora.train --dataset "$2" --questions v3 --train 500 --eval 100 --max_ctx 4096 \
    --batch 1 --accum 8 --out "${1}500v3" 2>&1 | grep --line-buffered -E "\[eval|\[sched|\[train\] step|Traceback|Error|memory" \
    | tee "results/train_${1}500v3.log"
  uv run python -m bench.lora.merge "adapters/${1}500v3" 2>&1 | tail -1
  arm "adapters/${1}500v3-merged" "decider-2b+${1}500" "$2"
  rm -rf "adapters/${1}500v3-merged"
}
echo "[stage] base, prompt-engineered"
arm Mapika/decider-2b decider-2b abcd
arm Mapika/decider-2b decider-2b actions
adapter B actions
adapter A abcd
echo "[v3 done]"
