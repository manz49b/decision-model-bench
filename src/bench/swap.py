"""Act 5 measured: one bf16 base with both 500-chat adapters attached (not merged), switching per job.

Measures what the "one base, many adapters" setup claims:
- accuracy on both jobs' test300, which should match the merged adapters up to bf16 rounding (anything more is a bug);
- latency per call with the adapter attached vs merged (an attached adapter adds work to every call);
- time to switch adapters, and GPU memory for one base plus two adapters.

Scores are logged to results/runs.csv as ``decider-2b+A500-unmerged`` / ``decider-2b+B500-unmerged``, with the same
dev-refitted temperatures as the merged arms.

    CC=.bin/cc uv run python -m bench.swap
"""

import json
import statistics
import time
from pathlib import Path

import torch
from decider.infer import Decider
from peft import PeftModel

from bench import run
from bench.data import dataset
from bench.engines import meta, normalise

ROOT = Path(__file__).parents[2]
JOBS = {"A": ("abcd", "decider-2b+A500"), "B": ("actions", "decider-2b+B500")}


class Swappable:
    """A decider engine over one base model whose active LoRA adapter can be switched."""

    def __init__(self):
        self.model = Decider("Mapika/decider-2b", use_graphs=False)
        # PEFT injects the LoRA layers into the base model's modules in place, so decider's own forward pass uses them
        self.peft = PeftModel.from_pretrained(self.model.m.lm, ROOT / "adapters" / "A500", adapter_name="A")
        self.peft.load_adapter(ROOT / "adapters" / "B500", adapter_name="B")
        self.peft.eval()
        self.temps = {job: json.loads((ROOT / f"results/temps_{label}_{ds}.json").read_text())
                      for job, (ds, label) in JOBS.items()}
        self.name, self.last_meta = "", {}

    def use(self, job: str) -> float:
        """Switch the active adapter (and that job's temperatures). Returns the switch time in ms."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.peft.set_adapter(job)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        self.model.T_by_type = self.temps[job]
        self.name = JOBS[job][1] + "-unmerged"
        return ms

    def decide(self, state: str, questions: dict) -> dict:
        body = self.model.system_one(state, questions)
        self.last_meta = {**meta(body), "model": f"{self.name} (base + attached adapter)"}
        return normalise(body["answers"])


def main() -> None:
    engine = Swappable()
    out = {"gpu": torch.cuda.get_device_name(0)}
    for job, (ds, _) in JOBS.items():
        engine.use(job)
        data = dataset(ds)
        for sample in ["test300"] + (["dev150"] if ds == "abcd" else []):
            df = run.run([engine], data.sample(sample), data.questions("v2"),
                         tags={"dataset": ds, "sample": sample, "question_set": "v2"})
            out[f"{job}/{sample}"] = {"run_id": df["run_id"].iat[0],
                                      "p50_ms": round(df.drop_duplicates("case_id")["latency_ms"].median(), 1)}
            print(job, sample, out[f"{job}/{sample}"], flush=True)
    switches = [engine.use("A" if i % 2 else "B") for i in range(200)]
    out["switch_ms"] = {"median": round(statistics.median(switches), 3), "max": round(max(switches), 3)}
    out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(out, flush=True)
    (ROOT / "results" / "swap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
