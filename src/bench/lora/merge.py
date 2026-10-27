"""Merge a LoRA adapter into decider's bf16 weights, so it serves exactly like a released decider model.

The adapter was trained on a 4-bit base; merging into bf16 is the standard QLoRA route (decider-2b v11 was
itself a merged LoRA). ``decider_config.json`` (temperatures, layout) is copied from the base model.

    uv run python -m bench.lora.merge adapters/lora500
"""

import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter: Path, base: str = "Mapika/decider-2b") -> Path:
    out = adapter.with_name(adapter.name + "-merged")
    lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    lm = PeftModel.from_pretrained(lm, adapter).merge_and_unload()
    lm.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    shutil.copy(hf_hub_download(base, "decider_config.json"), out / "decider_config.json")
    return out


if __name__ == "__main__":
    print("[merged]", merge(Path(sys.argv[1])))
