"""ABCD (ASAPP, MIT licence): customer↔agent chats, turned into typed questions with gold labels.

Download first (raw files are git-ignored):
    mkdir -p data/raw/abcd && curl -sSL -o data/raw/abcd/abcd_v1.1.json.gz \
        https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz

Questions:
- ``flow``: 10-way ``choice``. Generic intents, the case where out-of-the-box engines should do well.
- ``subflow``: 55-way ``choice`` with ABCD's own fine-grained labels. The niche case where LoRA should help.
- ``is_product_return``: ``noul``, true when the flow is ``product_defect`` (returns and refunds).

Names, emails and addresses in ABCD are synthetic (role-played by crowd workers), so no real customer data.
"""

import gzip
import json
from functools import cache
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parents[3] / "data" / "raw" / "abcd"
SAMPLES = Path(__file__).parents[3] / "data" / "samples"

FLOWS = {
    "account_access": "Can't get into their account: forgotten username or password, or a two-factor problem.",
    "manage_account": "Changing account details (address, name, phone, payment method) or a question about their account.",
    "order_issue": "A problem with or change to an existing order: fees, delivery time, quantity, upgrade, cancel.",
    "product_defect": "Returning an item or a refund: wrong size or colour, a stain, or the status of a refund.",
    "purchase_dispute": "Disputing a price, promo code, out-of-stock item or a charge for something returned or never bought.",
    "shipping_issue": "Where their package is, a missing package, changing shipping, or shipping cost.",
    "single_item_query": "A question about one specific product (boots, shirt, jeans or a jacket).",
    "storewide_query": "A general question about the store: pricing, membership, timing or policy.",
    "subscription_inquiry": "Their subscription: status, amount or date due, paying the bill, an extension, a disputed bill.",
    "troubleshoot_site": "The website isn't working: card errors, shopping cart, search results or slowness.",
}


@cache
def subflows() -> dict[str, list[str]]:
    return json.loads((RAW / "ontology.json").read_text())["intents"]["subflows"]


SUBFLOW_ALIASES = {"status_delivery_date": "status_delivery_time"}


def subflow_key(flow: str, subflow: str) -> str | None:
    """ABCD reuses some subflow names (``status``, ``manage``) across flows, so keys include the flow.

    About 22% of conversations use variants the ontology doesn't list (``pricing_3``, ``boots_how_2``): these map to
    the ontology label they extend. ``None`` for the one with no clear match (``status_questions``).
    """
    if flow == "storewide_query":
        subflow = subflow.rsplit("_", 1)[0] if subflow[-1].isdigit() else subflow
    elif flow == "single_item_query":
        subflow = subflow.split("_", 1)[0]
    subflow = SUBFLOW_ALIASES.get(subflow, subflow)
    return f"{flow}.{subflow}" if subflow in subflows()[flow] else None


def questions(version: str = "v1") -> dict:
    if version == "v2":
        from bench.data.abcd_v2 import questions_v2

        return questions_v2()
    subflow_criteria = {
        subflow_key(flow, sub): f"{flow.replace('_', ' ')}: {sub.replace('_', ' ')}"
        for flow, subs in subflows().items()
        for sub in subs
    }
    return {
        "flow": {"type": "choice", "instructions": "What is the customer contacting support about?", "criteria": FLOWS},
        "subflow": {
            "type": "choice",
            "instructions": "Which specific support case is this? Pick the closest match.",
            "criteria": subflow_criteria,
        },
        "is_product_return": {
            "type": "noul",
            "instructions": "Does the customer want to return an item or get a refund for it?",
        },
    }


def state(turns: list, max_turns: int | None = None) -> str:
    """The chat as plain text. Agent tool actions are dropped: they'd give the answer away."""
    lines = [f"{speaker.capitalize()}: {text}" for speaker, text in turns if speaker != "action"]
    return "\n".join(lines[:max_turns])


def load(split: str = "test", max_turns: int | None = None, limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    """One row per conversation: id, state, and one gold column per question name."""
    with gzip.open(RAW / "abcd_v1.1.json.gz") as f:
        convos = json.load(f)[split]
    df = pd.DataFrame([
        {
            "id": c["convo_id"],
            "state": state(c["original"], max_turns),
            "flow": c["scenario"]["flow"],
            "subflow": subflow_key(c["scenario"]["flow"], c["scenario"]["subflow"]),
            "is_product_return": c["scenario"]["flow"] == "product_defect",
        }
        for c in convos
    ]).dropna(subset=["subflow"])
    return df.sample(n=limit, random_state=seed).reset_index(drop=True) if limit else df


def freeze(name: str, split: str, n: int, seed: int = 0) -> None:
    """Write a fixed sample of conversation ids, committed to git, so every engine and every run sees the same cases."""
    SAMPLES.mkdir(parents=True, exist_ok=True)
    ids = load(split, limit=n, seed=seed)["id"]
    (SAMPLES / f"{name}.txt").write_text("\n".join(map(str, ids)) + "\n")


def sample(name: str, max_turns: int | None = None) -> pd.DataFrame:
    """A frozen sample: ``test300`` (the scored benchmark) or ``dev150`` (for tuning criteria, never ``test``)."""
    split = {"test": "test", "dev": "dev"}[name.rstrip("0123456789")]
    ids = [int(i) for i in (SAMPLES / f"{name}.txt").read_text().split()]
    return load(split, max_turns).set_index("id").loc[ids].reset_index()
