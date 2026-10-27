"""Job B: ABCD next agent action. Given the chat so far, which of the company's 30 actions does the agent take next?

ABCD labels every agent action turn (``delexed[i]["targets"][2]``) with a button from its ontology. One action
point is sampled per conversation (seeded by the conversation id), and the ``state`` is everything before it:
customer and agent turns, plus the earlier actions the agent already took (the agent knows its own history).

This is the company's procedure rather than the customer's intent, so it's the kind of label a general model
can't know from one-line descriptions. ``pull-up-account`` opens most chats, so the majority baseline is high:
report macro-F1 alongside accuracy.
"""

import gzip
import json
import random
from functools import cache

import pandas as pd

from bench.data.abcd import RAW, SAMPLES

V1 = {  # day-one one-liners, from the button names only
    "pull-up-account": "Pull up the customer's account.",
    "verify-identity": "Verify the customer's identity.",
    "validate-purchase": "Validate a purchase.",
    "ask-the-oracle": "Ask the oracle.",
    "make-password": "Make a new password.",
    "promo-code": "Give a promo code.",
    "subscription-status": "Check subscription status.",
    "offer-refund": "Offer a refund.",
    "make-purchase": "Make a purchase for the customer.",
    "record-reason": "Record the reason.",
    "enter-details": "Enter details.",
    "shipping-status": "Check shipping status.",
    "update-order": "Update the order.",
    "update-account": "Update the account.",
    "send-link": "Send a link.",
    "notify-team": "Notify a team.",
    "membership": "Check membership level.",
    "search-faq": "Search the FAQ.",
    "try-again": "Ask the customer to try again.",
    "log-out-in": "Ask the customer to log out and back in.",
    "instructions": "Give instructions.",
    "search-jeans": "Search the FAQ for jeans.",
    "search-shirt": "Search the FAQ for shirts.",
    "search-boots": "Search the FAQ for boots.",
    "search-jacket": "Search the FAQ for jackets.",
    "search-pricing": "Search the FAQ for pricing.",
    "search-timing": "Search the FAQ for timing.",
    "search-membership": "Search the FAQ for membership.",
    "search-policy": "Search the FAQ for policy.",
    "select-faq": "Select an FAQ answer.",
}


def questions(version: str = "v1") -> dict:
    if version == "v2":
        from bench.data.abcd_actions_v2 import INSTRUCTIONS, V2

        instructions, criteria = INSTRUCTIONS, V2
    else:
        instructions, criteria = "Which action should the support agent take next in this chat?", V1
    return {"next_action": {"type": "choice", "instructions": instructions, "criteria": criteria}}


@cache
def _convos(split: str) -> list:
    with gzip.open(RAW / "abcd_v1.1.json.gz") as f:
        return json.load(f)[split]


def _line(speaker: str, text: str) -> str:
    return f"[Agent action] {text}" if speaker == "action" else f"{speaker.capitalize()}: {text}"


def load(split: str = "test", limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    """One row per conversation: id, state (the chat before one sampled action), next_action (gold)."""
    rows = []
    for c in _convos(split):
        points = [i for i, t in enumerate(c["delexed"]) if t["speaker"] == "action"]
        i = random.Random(c["convo_id"]).choice(points)
        rows.append({
            "id": c["convo_id"],
            "state": "\n".join(_line(s, t) for s, t in c["original"][:i]),
            "next_action": c["delexed"][i]["targets"][2],
            "position": points.index(i),
        })
    df = pd.DataFrame(rows)
    return df.sample(n=limit, random_state=seed).reset_index(drop=True) if limit else df


def freeze(name: str, split: str, n: int, seed: int = 0) -> None:
    ids = load(split).sample(n=n, random_state=seed)["id"]
    (SAMPLES / f"actions_{name}.txt").write_text("\n".join(map(str, ids)) + "\n")


def sample(name: str) -> pd.DataFrame:
    split = {"test": "test", "dev": "dev"}[name.rstrip("0123456789")]
    ids = [int(i) for i in (SAMPLES / f"actions_{name}.txt").read_text().split()]
    return load(split).set_index("id").loc[ids].reset_index()
