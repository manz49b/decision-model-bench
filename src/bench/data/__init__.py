"""Datasets: each module has ``load(split, limit=, seed=)``, ``sample(name)`` and ``questions(version)``."""


def dataset(name: str):
    if name == "abcd":
        from bench.data import abcd as module
    elif name == "actions":
        from bench.data import abcd_actions as module
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return module
