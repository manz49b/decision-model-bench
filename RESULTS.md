# Results

Every number behind the write-up, with intervals and paired tests. The [README](README.md) has the summary and how to
run everything; `uv run python -m bench.story_table` rebuilds the two generated tables below from `results/runs.csv`.

## Prompt or fine-tune: Jev vs decider-2b

ABCD customer-support chats (about 10k, MIT licence). Two jobs, **the same company-written criteria for every engine**,
each scored on the same 300 held-out chats. "…" = not run (the 2,000-chat adapter was only trained for job B).

<!-- results:start -->
| Engine | A: `flow` (10) | A: `subflow` (54) | A: return yes/no (F1)³ | B: next action (30) | Speed, p50 per call (A / B) | Cost per 1,000 calls (A / B) |
|---|---|---|---|---|---|---|
| **Jev** (TypeSafe API) | 0.87 | 0.79 | 0.77 | 0.56 | 228 / 227 ms | $0.103 / $0.062 |
| **Jev, prompt-engineered** (+ 5 similar labelled chats) | 0.91 | 0.87 | 0.81 | 0.71 | 227 / 224 ms | $0.146 / $0.092 |
| **decider-2b** (open, no training) | 0.79 | 0.71 | 0.63 | 0.33 | 885 / 231 ms² | ~$0.033–0.098 / ~$0.019¹ |
| **decider-2b + adapter**, 100 chats | 0.83 | 0.79 | 0.85 | 0.55 | 874 / 229 ms² | ~$0.033–0.098 / ~$0.019¹ |
| **decider-2b + adapter**, 500 chats | 0.94 | 0.85 | **0.92** | **0.73** | 880 / 264 ms² | ~$0.033–0.098 / ~$0.019¹ |
| **decider-2b + adapter**, 2,000 chats (job B only) | … | … | … | 0.72 | — / 234 ms² | — / ~$0.019¹ |
| **decider-2b, prompt-engineered** (no training) | 0.79 | 0.72 | 0.57 | 0.27 | 1,656 / 395 ms² | ~$0.052–0.155 / ~$0.032¹ |
| **decider-2b + adapter**, 500 chats, **prompt-engineered** | **0.95** | **0.89** | 0.87 | 0.71 | 1,723 / 384 ms² | ~$0.052–0.155 / ~$0.032¹ |
<!-- results:end -->

¹ **Estimated, not measured: optimised self-hosting on an AWS g5.xlarge** (1× A10G, $1.006/h), kept fully busy.
GPU seconds per call = 2 FLOPs per parameter × 2B parameters × measured input tokens × forward passes, divided by
62.5 TFLOPS (the A10G's 125 TFLOPS bf16 peak at 50% utilisation, typical for batched prefill-only scoring). Job B is one
pass (about 0.07 s). Job A asks three questions: the **high end** pays a full pass for each (what the laptop's timings
imply, about 0.35 s), the **low end** shares the chat across them with prefix caching (about 0.12 s). Two conditions,
per job: **(1) cost per call:** job B is about 3× cheaper than Jev; job A is only cheaper with prefix caching, and
level with Jev without it. **(2) volume:** a dedicated GPU only pays off when it's busy: at least **2.7 / 4.5 req/s**
sustained against Jev (**1.9 / 3.0** against Jev + examples). Without prefix caching one A10G tops out near 2.9 job A
req/s, barely above break-even. The laptop itself can't stand in: batching gave no gain on it (job A 0.87 s per call
batched vs 0.88 s one at a time; `results/throughput.json`), and at its speed Jev is cheaper on every job. Confirming the
estimate needs a batched serving run (e.g. vLLM) on a cloud GPU.
² Laptop GPU (6 GB RTX PRO 500), unoptimised, one call at a time: not comparable with a hosted API. Job A calls carry three
questions and are about 4× slower than job B's. Speed and cost are shown per job because a median over both jobs mixed
together depends on which jobs an arm ran.
³ Each engine's yes/no threshold is the F1-best one on the dev chats, applied unchanged on test (a fixed 0.5 understates
well-calibrated models on a class that's only 10% of chats).

The best score in each column is **bold**. **How to read the gaps:** unpaired, n=300 gives intervals of about ±4–5 points,
but every engine answers the same chats, so the **paired** tests in the detail section detect smaller gaps. All 54 paired
tests are **Holm-corrected as one family** (family-wise 5%). In the text, **"beats" means it survives that correction**;
anything that's only p<0.05 on its own is called "suggestive".

**Job A, triage:** what does the customer want? `flow` = 10 broad intents, `subflow` = the company's 54 fine-grained
cases, return yes/no = does the customer want to return an item or get a refund for it.
**Job B, next action:** given the chat so far, which of the company's 30 playbook actions should the agent take next?

## Hosted decision APIs: Jev vs ai_decide vs OpenAI Decisions

Added 26–27 Oct 2026. Same 300 test chats per job and the same questions (`v2` criteria only, `v3` + 5 retrieved
examples), one call per chat with all its questions, no caching. Engines: `ai_decide` (Databricks SQL function, version
reported `ai_decide-1.0`) and `openai_decisions` (`gpt-6-luna`). Paired McNemar tests against Jev on the same chats.

| | Triage: `flow` | Triage: `subflow` | Next action |
|---|---|---|---|
| Jev | 0.87 | 0.79 | 0.56 |
| Jev + examples | 0.91 | 0.87 | **0.71** |
| OpenAI Decisions | 0.88 | 0.79 | 0.57 |
| OpenAI Decisions + examples | **0.92** | 0.86 | 0.63 |
| ai_decide | 0.86 | 0.70 | 0.39 |
| ai_decide + examples | 0.91 | 0.73 | 0.43 |

- **OpenAI Decisions vs Jev:** level on every question with criteria only (p ≥ 0.45). With examples, level on triage, but
  Jev is ahead on next action (0.71 vs 0.63, p = 0.002).
- **ai_decide vs Jev:** level on `flow`; clearly behind on `subflow` and next action, with and without examples
  (p < 0.0001).
- **What's behind ai_decide** (`scripts/ai_decide_check.py`, dev150, job A): same-answer rate with Jev 1.14 is 96% / 81% /
  96% (`flow` / `subflow` / return), against 98% / 93% / 99% between Jev 1.13 and 1.14. With Databricks' OpenJev
  (`system.ai.openjev-qwen35-4b`) it's 96% / 89% / 100%, with a mean yes/no probability gap of 0.005. Databricks' docs say
  ai_decide runs on Apache 2.0 models. So ai_decide behaves like OpenJev, not Jev, though it accepts Jev's API format.
- **Cost and speed:** OpenAI is $0.10 per million input tokens (output free), $0.11–0.20 per 1,000 calls ($0.17–0.30 with
  examples), about 300 ms p50. ai_decide bills as Databricks serverless compute: 3.76 DBU (about $3.58 at list price) for
  about 1,200 single calls at about 3 s each, so about $3 per 1,000 one call at a time. As one batch query
  (`scripts/ai_decide_batch.py`) the same 300 next-action chats took 19 s with identical answers.
- **Run notes:** some ai_decide calls hit a temporary Databricks "unauthorized network access" error; they were re-run with
  `scripts/ai_decide_retry.py`, and scoring drops errored rows, so every chat has one answer. These tests are reported
  uncorrected (not part of the 54-test Holm family).

## Speed and cost: Jev vs self-hosted decider-2b

| | Jev | decider-2b (+ adapters), self-hosted |
|---|---|---|
| Pricing model | $0.042 per 1M input tokens, output tokens free (docs.typesafe.ai/models, checked 26 Oct 2026; same for Jev 1.13 and 1.14) | GPU time, priced here at $1.006/h for an AWS g5.xlarge (third-party price list, Jul 2026) |
| Tokens per call (measured) | 2,452 (job A, 3 questions) and 1,485 (job B); with examples 3,479 and 2,181 | the same, but no per-token charge |
| Speed (p50, job A / B) | about 228 / 227 ms over the internet from the UK, the same with examples | 880 / 230 ms on a 6 GB laptop GPU (1,720 / 390 ms with examples), unoptimised, one call at a time |
| Cost at volume | linear: $0.103 / $0.062 per 1,000 calls (job A / B); $0.146 / $0.092 with examples | fixed GPU cost. Estimated on an optimised A10G at full use: $0.033–0.098 / $0.019 per 1,000 (job A low end needs prefix caching; $0.052–0.155 / $0.032 with examples). FLOPs estimate, footnote ¹. Cheaper than Jev only above 2.7 / 4.5 req/s sustained (1.9 / 3.0 against Jev + examples). On this laptop (batching measured, no gain) it costs more than Jev |
| Fine-tuning | No: the same weights serve every account | Yes: one 67 MB adapter per job, about 35 min for 500 chats on the laptop |
| Data stays… | with TypeSafe: not trained on customer requests; zero data retention offered to enterprise customers, terms via TypeSafe sales (docs.typesafe.ai/models and /legal, checked 26 Oct 2026) | on your own hardware |

## What I didn't test, and why

- **OpenJev, Hugging Face builds (not tested).** The Databricks build is covered through ai_decide (see above). It's an open, Jev-compatible model in three builds: Databricks-hosted
  `system.ai.openjev-qwen35-4b`, and Hugging Face `openjev/openjev` (27B) and `OpenJev-Flash-9B`. I left it out because the
  Hugging Face weights are **CC BY-NC 4.0** (non-commercial; commercial use needs a licence from Loop AI), no training code is
  published, the 27B doesn't fit a laptop, and the Databricks build needs a workspace login. Nothing shows the three builds
  share weights. **Assumed, not measured:** used as it is, it should land between decider-2b and Jev, since it's a bigger
  open model trained for the same System One task. Fine-tuned, a bigger base would likely start higher than decider-2b and
  need fewer labelled chats to pass Jev, at a higher GPU cost to train and serve.
- **Fine-tuning Jev.** Not possible: TypeSafe serves the same weights to every account. Fine-tuning needs open weights
  (decider here, Apache 2.0; or OpenJev under its licence).
- **Optimised serving.** decider ran unoptimised on a 6 GB laptop. Batched throughput on a cloud GPU (the basis for the
  self-hosted cost estimate, footnote ¹) wasn't measured.

## Caveats

- **Role-play data:** some gold labels are noisy, so nothing reaches 100%.
- **One set of criteria, written by me** from the company guidelines: one pass checked on dev chats, then frozen. Dev
  scores ran a few points above test, so there's a small optimism in any criteria tuning.
- **The criteria and the example prompt were tuned on Jev's dev errors**, not decider's. Any bias from that favours Jev,
  so the decider results are, if anything, understated.
- **Training-run variation:** the job B 500-chat adapter, retrained with 3 seeds on the same chats, spans 0.717–0.737
  (2.0 points), so job B methods within that range are tied. On job A, three seeds per 500-chat variant span 1.7 points
  on `flow` and 0.7–1.4 points on `subflow`. Seed-to-seed thresholds on the return yes/no range from 0.10 to 1.00 (dev
  scores are near-saturated), which is another reason to read average precision rather than F1 there.
- **The return yes/no** has about 30 positives in 300 chats. F1 swings with the threshold, so average precision is the steadier comparison.
- **n=300 per job:** unpaired intervals are about ±4–5 points. Paired tests detect smaller gaps. All 54 are Holm-corrected,
  and the text only says "beats" for those that survive.
- **Latency isn't like-for-like** (hosted API vs laptop GPU), and cost for self-hosting depends on your hardware.
- **A technical comparison on one public dataset,** not a production recommendation.

## Detail: every score with its interval and a paired test against Jev

<!-- detail:start -->
54 paired tests in this section. ✓ Holm = still significant after Holm correction across all 54 (family-wise 5%); "not after Holm" = p<0.05 on its own only, so treat as suggestive.

| Engine | Test | Score | 95% interval | vs Jev, criteria only (paired p) | vs Jev + examples (paired p) | Yes/no: threshold (from dev) · average precision |
|---|---|---|---|---|---|---|
| Jev (TypeSafe API) | A: `flow` (accuracy) | 0.870 | 0.83–0.91 | — | 0.002 (worse, ✓ Holm) |  |
| Jev (TypeSafe API) | A: `subflow` (accuracy) | 0.790 | 0.75–0.83 | — | <0.001 (worse, ✓ Holm) |  |
| Jev (TypeSafe API) | A: `is_product_return` (F1) | 0.771 | 0.63–0.88 | — | 0.25 (no difference) | 0.79 · 0.87 |
| Jev (TypeSafe API) | B: `next_action` (accuracy) | 0.556 | 0.50–0.61 | — | <0.001 (worse, ✓ Holm) |  |
| Jev, prompt-engineered (+ 5 similar labelled chats) | A: `flow` (accuracy) | 0.912 | 0.88–0.94 | 0.002 (better, ✓ Holm) | — |  |
| Jev, prompt-engineered (+ 5 similar labelled chats) | A: `subflow` (accuracy) | 0.874 | 0.83–0.91 | <0.001 (better, ✓ Holm) | — |  |
| Jev, prompt-engineered (+ 5 similar labelled chats) | A: `is_product_return` (F1) | 0.814 | 0.69–0.91 | 0.25 (no difference) | — | 0.69 · 0.91 |
| Jev, prompt-engineered (+ 5 similar labelled chats) | B: `next_action` (accuracy) | 0.708 | 0.66–0.76 | <0.001 (better, ✓ Holm) | — |  |
| decider-2b (open, no training) | A: `flow` (accuracy) | 0.790 | 0.74–0.84 | <0.001 (worse, ✓ Holm) | <0.001 (worse, ✓ Holm) |  |
| decider-2b (open, no training) | A: `subflow` (accuracy) | 0.710 | 0.66–0.76 | <0.001 (worse, ✓ Holm) | <0.001 (worse, ✓ Holm) |  |
| decider-2b (open, no training) | A: `is_product_return` (F1) | 0.627 | 0.47–0.75 | 0.012 (worse, not after Holm) | <0.001 (worse, ✓ Holm) | 0.29 · 0.67 |
| decider-2b (open, no training) | B: `next_action` (accuracy) | 0.330 | 0.28–0.38 | <0.001 (worse, ✓ Holm) | <0.001 (worse, ✓ Holm) |  |
| decider-2b + adapter, 100 chats | A: `flow` (accuracy) | 0.833 | 0.79–0.87 | 0.09 (no difference) | <0.001 (worse, ✓ Holm) |  |
| decider-2b + adapter, 100 chats | A: `subflow` (accuracy) | 0.787 | 0.74–0.83 | 0.77 (no difference) | <0.001 (worse, ✓ Holm) |  |
| decider-2b + adapter, 100 chats | A: `is_product_return` (F1) | 0.852 | 0.73–0.94 | 0.33 (no difference) | 0.79 (no difference) | 0.11 · 0.90 |
| decider-2b + adapter, 100 chats | B: `next_action` (accuracy) | 0.547 | 0.49–0.60 | 0.73 (no difference) | <0.001 (worse, ✓ Holm) |  |
| decider-2b + adapter, 500 chats | A: `flow` (accuracy) | 0.937 | 0.91–0.96 | 0.001 (better, ✓ Holm) | 0.23 (no difference) |  |
| decider-2b + adapter, 500 chats | A: `subflow` (accuracy) | 0.847 | 0.81–0.89 | 0.020 (better, not after Holm) | 0.18 (no difference) |  |
| decider-2b + adapter, 500 chats | A: `is_product_return` (F1) | 0.915 | 0.82–0.98 | 0.039 (better, not after Holm) | 0.18 (no difference) | 0.10 · 0.96 |
| decider-2b + adapter, 500 chats | B: `next_action` (accuracy) | 0.733 | 0.68–0.78 | <0.001 (better, ✓ Holm) | 0.71 (no difference) |  |
| decider-2b + adapter, 2,000 chats (job B only) | B: `next_action` (accuracy) | 0.720 | 0.67–0.77 | <0.001 (better, ✓ Holm) | 1.00 (no difference) |  |
| decider-2b, prompt-engineered (no training) | A: `flow` (accuracy) | 0.787 | 0.74–0.83 | <0.001 (worse, ✓ Holm) | <0.001 (worse, ✓ Holm) |  |
| decider-2b, prompt-engineered (no training) | A: `subflow` (accuracy) | 0.720 | 0.67–0.77 | <0.001 (worse, ✓ Holm) | <0.001 (worse, ✓ Holm) |  |
| decider-2b, prompt-engineered (no training) | A: `is_product_return` (F1) | 0.571 | 0.39–0.72 | 0.007 (worse, not after Holm) | <0.001 (worse, ✓ Holm) | 0.46 · 0.72 |
| decider-2b, prompt-engineered (no training) | B: `next_action` (accuracy) | 0.267 | 0.22–0.32 | <0.001 (worse, ✓ Holm) | <0.001 (worse, ✓ Holm) |  |
| decider-2b + adapter, 500 chats, prompt-engineered | A: `flow` (accuracy) | 0.950 | 0.92–0.97 | <0.001 (better, ✓ Holm) | 0.007 (better, not after Holm) |  |
| decider-2b + adapter, 500 chats, prompt-engineered | A: `subflow` (accuracy) | 0.887 | 0.85–0.92 | <0.001 (better, ✓ Holm) | 0.48 (no difference) |  |
| decider-2b + adapter, 500 chats, prompt-engineered | A: `is_product_return` (F1) | 0.867 | 0.76–0.95 | 0.27 (no difference) | 0.77 (no difference) | 0.77 · 0.96 |
| decider-2b + adapter, 500 chats, prompt-engineered | B: `next_action` (accuracy) | 0.713 | 0.66–0.77 | <0.001 (better, ✓ Holm) | 0.90 (no difference) |  |

**Does training on example-enriched prompts beat the plain 500-chat adapter?** (paired p)

| Test | p |
|---|---|
| A: `flow` | 0.45 (no difference) |
| A: `subflow` | 0.017 (better, not after Holm) |
| A: `is_product_return` | 0.38 (no difference) |
| B: `next_action` | 0.42 (no difference) |

**Right vs wrong adapter** (500 chats each; paired p vs the base model):

| | A: `subflow` | B: next action |
|---|---|---|
| base decider-2b | 0.71 | 0.33 |
| adapter A (triage) | 0.85, p <0.001 (better, ✓ Holm) | 0.30, p 0.33 (no difference) |
| adapter B (next action) | 0.71, p 1.00 (no difference) | 0.73, p <0.001 (better, ✓ Holm) |
<!-- detail:end -->

## How it was run

- **Engines:** Jev = TypeSafe System One API, `jev-latest` → `jev-1.13.0` (logged per call). Jev 1.14 was released afterwards and wasn't tested. decider-2b = Mapika's
  Apache-2.0 System One–style model (Qwen3.5 2B base). Adapters = QLoRA (4-bit, r=16, all linear layers, one epoch),
  merged into bf16 for scoring.
- **Fairness:** the same frozen test sample and the same question text for every engine. Jev was run 3× per chat (98–99%
  identical answers). Its accuracy and intervals use all three runs (each chat's runs averaged, resampled by chat), and
  the paired tests use the first run. Repeating the key paired tests with Jev's majority answer across the three runs
  changes no conclusion (largest shift: adapter vs Jev on `subflow`, p 0.020→0.009, still not after Holm). decider is deterministic, so it ran once. Every decider arm, base included, has its temperatures
  refitted on that job's dev chats, because decider's shipped yes/no temperature was about 50× too sharp for this data.
- **Contamination:** ABCD isn't in decider's training mix. Banking77, CLINC, Bitext and others are, which is why they
  aren't used.
