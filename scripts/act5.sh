#!/usr/bin/env bash
# Act 5 measured, after the seed runs: (1) one shared adapter for both jobs; (2) one base with both adapters attached.
set -euo pipefail
cd "$(dirname "$0")/.."
export CC="$PWD/.bin/cc"
until grep -q "\[seeds done\]" results/seeds.log 2>/dev/null; do sleep 60; done
quiet() { grep --line-buffered -v -i "warn\|causal_conv1d\|Loading weights"; }
echo "[stage] shared adapter (both jobs, 500 chats each)"
if [ ! -f adapters/AB500/adapter_model.safetensors ]; then
  uv run python -m bench.lora.train --dataset both --questions v2 --train 500 --eval 100 --out AB500 2>&1 \
    | grep --line-buffered -E "\[eval|\[sched|\[data|Traceback|Error|memory"
fi
uv run python -m bench.lora.merge adapters/AB500 2>&1 | tail -1
for ds in abcd actions; do
  temps="results/temps_decider-2b+AB500_${ds}.json"
  uv run python -m bench.lora.calibrate adapters/AB500-merged "$temps" v2 "$ds" 2>&1 | grep "\[calibrated\]"
  for sample in test300 $([ "$ds" = abcd ] && echo dev150); do
    uv run python -m bench.run --engine decider --path adapters/AB500-merged --label decider-2b+AB500 --temps "$temps" \
      --dataset "$ds" --sample "$sample" --questions v2 2>&1 | quiet | tail -1
  done
done
rm -rf adapters/AB500-merged
echo "[stage] swap test (one base, both adapters attached)"
uv run python -m bench.swap 2>&1 | quiet
echo "[act5 done]"
