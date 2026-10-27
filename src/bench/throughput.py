"""Measured batched throughput of decider-2b on this machine's GPU, per job and input.

Replaces the FLOPs estimate behind the self-hosting break-even. Uses decider's own System One scoring path with many
chats packed into each forward pass (the one-at-a-time harness latency is the worst case). An adapter is the same
architecture once merged, so the base model's throughput applies to every decider arm.

    CC=.bin/cc uv run python -m bench.throughput
"""

import json
import time
from pathlib import Path

import torch
from decider import temperature as TT
from decider.infer import Decider

from bench.data import dataset
from bench.prompting import Prompted

OUT = Path(__file__).parents[2] / "results" / "throughput.json"
N = 96


def measure(model: Decider, states: list[str], questions: dict, max_fwd_tokens: int) -> dict:
    items = [it for s in states for it in model._system_one_items(s, questions)[2]]

    def score(batch):
        return model._system_one_probs(batch, "state_first", max_fwd_tokens, TT.for_items(model.T, model.T_by_type, batch))

    score(items[:8])  # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    score(items)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return {"calls": len(states), "rows": len(items), "seconds": round(seconds, 2),
            "calls_per_s": round(len(states) / seconds, 2), "max_fwd_tokens": max_fwd_tokens,
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def main() -> None:
    model = Decider("Mapika/decider-2b", use_graphs=False)
    results = {}
    for name, version in [("abcd", "v2"), ("actions", "v2"), ("abcd", "v3"), ("actions", "v3")]:
        data = Prompted(dataset(name), name) if version == "v3" else dataset(name)
        states = data.sample("test300")["state"].head(N).tolist()
        questions = data.questions(version)
        for fwd in (8192, 16384):
            try:
                r = measure(model, states, questions, fwd)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                r = {"max_fwd_tokens": fwd, "error": "out of memory"}
            results[f"{name}/{version}/{fwd}"] = r
            print(name, version, r, flush=True)
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "results": results}, indent=1))


if __name__ == "__main__":
    main()
