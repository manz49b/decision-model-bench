#!/usr/bin/env bash
# Seed variation (review item 5): two more training runs (seeds 1, 2) of A500, A500 + examples and B500, on the same 500
# chats (data_seed 0). Each is calibrated on dev and scored on test, labelled e.g. decider-2b+A500s1.
set -euo pipefail
cd "$(dirname "$0")/.."
export CC="$PWD/.bin/cc"
quiet() { grep --line-buffered -v -i "warn\|causal_conv1d\|Loading weights"; }
one() {  # job dataset questions seed
  local job=$1 ds=$2 qs=$3 seed=$4 extra=""
  local name="${job}500$([ "$qs" = v3 ] && echo v3)s${seed}" label="decider-2b+${job}500s${seed}"
  [ "$qs" = v3 ] && extra="--max_ctx 4096 --batch 1 --accum 8"
  echo "[stage] $name"
  if uv run python -c "import pandas as pd,sys; d=pd.read_csv('results/runs.csv',low_memory=False); sys.exit(0 if ((d.engine=='$label')&(d.question_set=='$qs')&(d['sample']=='test300')).any() else 1)" 2>/dev/null; then
    echo "[skip] $name already scored"; return
  fi
  if [ -f "adapters/$name/adapter_model.safetensors" ]; then echo "[resume] $name already trained"; else
  uv run python -m bench.lora.train --dataset "$ds" --questions "$qs" --train 500 --eval 100 --seed "$seed" $extra \
    --out "$name" 2>&1 | grep --line-buffered -E "\[eval after|\[sched|Traceback|Error|memory"
  fi
  uv run python -m bench.lora.merge "adapters/$name" 2>&1 | tail -1
  local temps="results/temps_${label}_${ds}_${qs}.json"
  uv run python -m bench.lora.calibrate "adapters/$name-merged" "$temps" "$qs" "$ds" 2>&1 | grep "\[calibrated\]"
  for sample in test300 $([ "$ds" = abcd ] && echo dev150); do
    uv run python -m bench.run --engine decider --path "adapters/$name-merged" --label "$label" --temps "$temps" \
      --dataset "$ds" --sample "$sample" --questions "$qs" 2>&1 | quiet | tail -1
  done
  rm -rf "adapters/$name-merged"
}
for seed in 1 2; do one B actions v2 $seed; done
for seed in 1 2; do one A abcd v2 $seed; done
for seed in 1 2; do one A abcd v3 $seed; done
echo "[seeds done]"
