"""Question set v3: prompt engineering on the *input*, applied identically to every engine.

**Final v3 = v2 criteria + retrieved examples.** Chosen on dev150 with Jev: examples drove all the
gains, while the two-step subflow and the playbook each hurt on top of them, so both are off by default (kept as
ablation switches).

On top of the v2 criteria:
- **Retrieved examples** (both jobs): the k most similar labelled chats from a pool of training chats, with their
  correct answers, are put in ``state``. The pool is the same 500 training chats the 500-chat adapters learned
  from, so prompting and fine-tuning get the same labelled-data budget. TF-IDF similarity, no extra model.
- **Playbook** (job B): the company's procedure, one line of actions per case, condensed from ABCD's guidelines.
- **Two-step subflow** (job A, see ``TwoStep``): ask ``flow`` first, then choose ``subflow`` only among that flow's
  cases (3–8 instead of 54).

The chat itself always comes first in ``state``, so nothing truncates it.
"""

import json
from functools import cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from bench.data import abcd
from bench.data.abcd import RAW

POOL_SIZE = 500
K = 5
EXAMPLE_CHARS = 700  # per example; job B keeps the end of the chat (where the decision is), job A the start

BUTTONS = {
    "Pull up Account": "pull-up-account", "Verify Identity": "verify-identity", "Validate Purchase": "validate-purchase",
    "Ask the Oracle": "ask-the-oracle", "Make Password": "make-password", "Promo Code": "promo-code",
    "Subscription Status": "subscription-status", "Offer Refund": "offer-refund", "Make Purchase": "make-purchase",
    "Record Reason": "record-reason", "Enter Details": "enter-details", "Enter Detail": "enter-details",
    "Shipping Status": "shipping-status", "Update Order": "update-order", "Update Account": "update-account",
    "Send Link": "send-link", "Notify Internal Team": "notify-team", "Membership Privileges": "membership",
    "Search FAQ": "search-faq", "Try Again": "try-again", "Log Out/In": "log-out-in", "Instructions": "instructions",
    "Boots": "search-boots", "Shirt": "search-shirt", "Jeans": "search-jeans", "Jacket": "search-jacket",
    "Pricing": "search-pricing", "Timing": "search-timing", "Membership": "search-membership", "Policy": "search-policy",
    "Select Answer": "select-faq",
}


@cache
def playbook() -> str:
    """The company procedure: for each case, the order its actions are taken in."""
    g = json.loads((RAW / "guidelines.json").read_text())
    lines = ["Company playbook (the usual order of agent actions for each kind of case):"]
    for flow, fv in g.items():
        for sub, sv in fv["subflows"].items():
            seq = []
            for a in sv["actions"]:
                b = BUTTONS.get(a.get("button"))
                if b and (not seq or seq[-1] != b):
                    seq.append(b)
            lines.append(f"- {flow} / {sub}: {' → '.join(seq)}")
    return "\n".join(lines)


class Retriever:
    """The K most similar labelled training chats for a new chat, formatted with their correct answers."""

    def __init__(self, data, questions: dict, pool_size: int = POOL_SIZE, seed: int = 0, tail: bool = False):
        self.pool = data.load("train", limit=pool_size, seed=seed)
        self.questions, self.tail = questions, tail
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english")
        self.matrix = self.vec.fit_transform(self.pool["state"])

    def _answer(self, row) -> str:
        parts = []
        for name, spec in self.questions.items():
            if spec["type"] == "noul":
                parts.append(f"{name} = {'yes' if row[name] else 'no'}")
            else:
                parts.append(f"{name} = {row[name]}")
        return "; ".join(parts)

    def examples(self, state: str, k: int = K, exclude_id=None) -> str:
        """``exclude_id``: leave-one-out, so a training chat that's in the pool never retrieves itself."""
        sims = linear_kernel(self.vec.transform([state]), self.matrix).ravel()
        order = [i for i in sims.argsort()[::-1] if self.pool["id"].iat[i] != exclude_id][:k]
        blocks = []
        for n, i in enumerate(order, 1):
            row = self.pool.iloc[i]
            chat = row["state"][-EXAMPLE_CHARS:] if self.tail else row["state"][:EXAMPLE_CHARS]
            blocks.append(f"Example {n}:\n{chat}\nCorrect answer: {self._answer(row)}")
        return "Similar past chats, with the correct answers:\n\n" + "\n\n".join(blocks)


class Prompted:
    """A dataset whose ``state`` is the v3 input: chat first, then playbook (job B) and retrieved examples."""

    def __init__(self, data, name: str, examples: bool = True, use_playbook: bool = False):
        self.data, self.name = data, name
        self.use_examples, self.use_playbook = examples, use_playbook and name == "actions"
        self.retriever = Retriever(data, data.questions("v2"), tail=(name == "actions"))

    def questions(self, version: str = "v3") -> dict:
        return self.data.questions("v2")

    def _v3(self, df: pd.DataFrame) -> pd.DataFrame:
        def build(s, case_id):
            parts = [f"The chat:\n{s}"]
            if self.use_playbook:
                parts.append(playbook())
            if self.use_examples:
                parts.append(self.retriever.examples(s, exclude_id=case_id))
            return "\n\n".join(parts)

        return df.assign(state=[build(s, i) for s, i in zip(df["state"], df["id"])])

    def load(self, split: str = "test", limit: int | None = None, seed: int = 0) -> pd.DataFrame:
        # Training on v3: the same 500 chats as the example pool (load("train", limit=500, seed=0) is the pool),
        # with leave-one-out retrieval, so every arm uses exactly the same 500 labelled chats
        return self._v3(self.data.load(split, limit=limit, seed=seed))

    def sample(self, name: str) -> pd.DataFrame:
        return self._v3(self.data.sample(name))


class TwoStep:
    """Wraps an engine: answer every question once, then re-ask ``subflow`` among the predicted flow's cases only."""

    def __init__(self, engine):
        self.engine, self.name = engine, engine.name
        self.subflows = abcd.subflows()

    def decide(self, state: str, questions: dict) -> dict:
        first = {k: v for k, v in questions.items() if k != "subflow"}
        answers = self.engine.decide(state, first)
        meta = dict(getattr(self.engine, "last_meta", {}))
        flow = answers["flow"]["label"]
        all_sub = questions["subflow"]["criteria"]
        keys = [abcd.subflow_key(flow, s) for s in self.subflows.get(flow, [])]
        keys = [k for k in keys if k in all_sub]
        if len(keys) >= 2:
            narrow = {"subflow": {**questions["subflow"], "criteria": {k: all_sub[k] for k in keys}}}
            answers |= self.engine.decide(state, narrow)
        elif len(keys) == 1:
            answers["subflow"] = {"label": keys[0], "score": answers["flow"]["score"]}
        second = getattr(self.engine, "last_meta", {})
        tokens = [m.get("input_tokens") for m in (meta, second) if m.get("input_tokens")]
        self.last_meta = {**second, "input_tokens": sum(tokens) if tokens else None}
        return answers
