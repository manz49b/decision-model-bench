"""LoRA fine-tune of decider on our typed questions, reusing decider's own prompt layout and loss.

decider has no separate heads: each answer is read from the LM's ``lm_head`` rows for the option letters at an
answer slot. So LoRA on the backbone's linear layers is enough, with nothing in ``modules_to_save``.

    uv run python -m bench.lora.train --train 400 --steps 50 --eval 100       # spike
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from decider.data.core import Example, Q
from decider.model import cap_logits, collate, softcap_value
from decider.prompt import MAX_OPTIONS, build, letter_ids
from decider.systemone import render_question
from decider.train import loss_fn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from bench.data import dataset

ADAPTERS = Path(__file__).parents[3] / "adapters"


def to_examples(row, questions: dict) -> list[Example]:
    """One decider Example per question, as decider's server lays them out (``independent=True``: each question in its
    own prompt over the same state). Training in a different layout from serving would quietly cost accuracy."""
    out = []
    for name, spec in questions.items():
        r = render_question(spec)
        gold = r["names"].index(bool(row[name]) if r["type"] == "noul" else row[name])
        out.append(Example(context=row["state"], qs=[Q(text=r["question"], options=r["options"], gold=gold)], task=name))
    return out


class Decider(torch.nn.Module):
    """decider's DecisionModel, but with a 4-bit or bf16 backbone we load ourselves so PEFT can wrap it."""

    def __init__(self, lm, tok):
        super().__init__()
        self.lm, self.tok = lm, tok
        self.register_buffer("letters", torch.tensor(letter_ids(tok)), persistent=False)
        self.softcap = softcap_value(lm.config)

    def forward(self, b):
        h = self.lm.get_base_model().model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        hs = h[b["slot_batch"], b["slot_idx"]]
        logits = cap_logits(self, F.linear(hs, self.lm.get_base_model().lm_head.weight[self.letters]).float())
        ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
        return logits.masked_fill(ar >= b["nopts"][:, None], float("-inf"))


def load(model_name: str, bits: int, r: int, alpha: int):
    tok = AutoTokenizer.from_pretrained(model_name)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    lm = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, quantization_config=quant if bits == 4 else None, device_map={"": 0}
    )
    if bits == 4:
        lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True)
    else:
        lm.gradient_checkpointing_enable()
    lm.enable_input_require_grads()
    lm = get_peft_model(lm, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules="all-linear"))
    lm.print_trainable_parameters()
    return Decider(lm, tok), tok


def batches(items, batch_size, rng):
    order = list(range(len(items)))
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


@torch.no_grad()
def evaluate(model, items, names, batch_size) -> dict:
    """Accuracy per question (argmax over the options; temperature doesn't change the argmax)."""
    model.eval()
    hits = {n: [] for n in names}
    for i in range(0, len(items), batch_size):
        b = collate(items[i : i + batch_size], model.tok.pad_token_id)
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        pred = model(b).argmax(-1).cpu()
        for p, g, name in zip(pred, b["golds"].cpu(), b["tasks"]):
            hits[name].append(int(p == g))
    model.train()
    return {n: round(sum(h) / len(h), 3) for n, h in hits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mapika/decider-2b")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 16])
    ap.add_argument("--train", type=int, default=400, help="training conversations")
    ap.add_argument("--eval", type=int, default=100, help="dev conversations")
    ap.add_argument("--steps", type=int, default=0, help="0 = one epoch over the training rows")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--out", default="spike")
    ap.add_argument("--seed", type=int, default=0, help="training randomness: LoRA init, dropout, batch order")
    ap.add_argument("--data_seed", type=int, default=0, help="which training chats; fixed at 0 so seed runs share the same 500")
    ap.add_argument("--max_ctx", type=int, default=1536, help="state tokens kept (v3 inputs need about 4096)")
    ap.add_argument("--questions", default="v1")
    ap.add_argument("--dataset", default="abcd", choices=["abcd", "actions", "both"])
    ap.add_argument("--brier_w", type=float, default=0.0, help="Brier term in the loss, as in decider's own training")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    # --dataset both: one shared adapter for both jobs, the same 500 chats per job as each specialist (1,000 in total).
    # No task tag: every row is one question with its own text and options, so the question says which job it is.
    jobs = ["abcd", "actions"] if a.dataset == "both" else [a.dataset]
    datas = {}
    for job in jobs:
        data = dataset(job)
        if a.questions == "v3":  # v3 inputs: chat + playbook + retrieved examples; the questions are v2's
            from bench.prompting import Prompted

            data = Prompted(data, job)
        datas[job] = data
    names = [n for job in jobs for n in datas[job].questions(a.questions)]
    model, tok = load(a.model, a.bits, a.r, a.alpha)
    model.cuda()

    # max_options=MAX_OPTIONS keeps all 55 subflows in every prompt, in a shuffled order
    def items(cases, questions):
        out = []
        for _, row in cases.iterrows():
            for ex in to_examples(row, questions):
                out.append({**build(ex, tok, rng, max_options=MAX_OPTIONS, max_ctx_tokens=a.max_ctx), "task": ex.task})
        return out

    # Train chats never overlap the frozen dev150 / test300 samples (different ABCD splits)
    train, dev = [], []
    for job, data in datas.items():
        q = data.questions(a.questions)
        train += items(data.load("train", limit=a.train, seed=a.data_seed), q)
        dev += items(data.sample("dev150").head(a.eval), q)
    print(f"[data] {len(train)} train rows, {len(dev)} dev rows, mean {sum(len(i['ids']) for i in train) / len(train):.0f} tokens")

    log = {"args": vars(a), "before": evaluate(model, dev, names, a.batch * 2)}
    print("[eval before]", log["before"])

    if not a.steps:
        a.steps = max(1, len(train) // (a.batch * a.accum))
    print(f"[sched] {a.steps} optimiser steps")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    step, micro, t0, losses = 0, 0, time.time(), []
    torch.cuda.reset_peak_memory_stats()
    while step < a.steps:
        for idx in batches(train, a.batch, rng):
            b = collate([train[i] for i in idx], tok.pad_token_id)
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            loss, ce = loss_fn(model(b), b["golds"], b["nopts"], brier_w=a.brier_w)
            (loss / a.accum).backward()
            losses.append(ce.item())
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"[train] step {step}/{a.steps} ce {sum(losses[-a.accum * 10:]) / (a.accum * 10):.3f} "
                      f"{time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
            if step >= a.steps:
                break

    log["after"] = evaluate(model, dev, names, a.batch * 2)
    log["train_seconds"] = round(time.time() - t0)
    log["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print("[eval after]", log["after"])
    out = ADAPTERS / a.out
    model.lm.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("[saved]", out)


if __name__ == "__main__":
    main()
