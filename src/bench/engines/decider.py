"""decider: Mapika's open System One model, run in-process on the local GPU, with or without a merged LoRA.

Uses ``decider.infer.Decider.system_one``: the same scoring path as decider's ``/v1/systemone`` server (one prompt
per question, options in the given order, per-type temperatures from ``decider_config.json``). The server itself
isn't used: its start-up captures CUDA graphs for batches up to 32, which thrashes a 6 GB GPU.
"""

from bench.engines import meta, normalise


class Decider:
    def __init__(self, path: str = "Mapika/decider-2b", name: str = "decider-2b", temperature_by_type: dict | None = None):
        """``temperature_by_type`` overrides decider_config.json, e.g. temperatures refitted on dev (bench.lora.calibrate)."""
        from decider.infer import Decider as Model

        self.name = name
        self.model = Model(path, use_graphs=False, temperature_by_type=temperature_by_type)
        self.last_meta = {"model": f"{name} ({path})", "input_tokens": None, "output_tokens": None}

    def decide(self, state: str, questions: dict) -> dict:
        body = self.model.system_one(state, questions)
        self.last_meta = {**meta(body), "model": self.last_meta["model"]}
        return normalise(body["answers"])
