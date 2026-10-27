# jev_engines_bench

Typed decision models compared on the same customer-support chats, with the same questions and paired significance tests:

- **Hosted decision APIs:** [Jev](https://docs.typesafe.ai/concepts/system-one) (TypeSafe AI), Databricks' [ai_decide](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_decide) and OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions).
- **Prompt or fine-tune:** [decider-2b](https://huggingface.co/Mapika/decider-2b) out of the box, with retrieved examples in the prompt, and with one QLoRA adapter per job, against Jev.

## Results at a glance

Accuracy on 300 unseen test chats per job. "+ examples" adds the five most similar labelled chats to each prompt; "fine-tuned" is one QLoRA adapter trained on 500 labelled chats.

| | Triage: broad intent (10) | Triage: exact case (54) | Next action (30) |
|---|---|---|---|
| Jev | 0.87 | 0.79 | 0.56 |
| Jev + examples | 0.91 | 0.87 | 0.71 |
| OpenAI Decisions | 0.88 | 0.79 | 0.57 |
| OpenAI Decisions + examples | 0.92 | 0.86 | 0.63 |
| ai_decide | 0.86 | 0.70 | 0.39 |
| ai_decide + examples | 0.91 | 0.73 | 0.43 |
| decider-2b | 0.79 | 0.71 | 0.33 |
| decider-2b, fine-tuned | 0.94 | 0.85 | 0.73 |
| decider-2b, fine-tuned + examples | 0.95 | 0.89 | 0.71 |

In short: Jev and OpenAI's Decisions API are level out of the box, and Jev gains more from examples. ai_decide behaves like Databricks' open OpenJev model rather than Jev, and trails on fine-grained questions. A fine-tuned 2B model catches up with the best hosted option. **[RESULTS.md](RESULTS.md)** has every score with its interval, the paired significance tests, the ai_decide agreement test, and the cost and speed numbers.

## The data and the jobs

The [Action-Based Conversations Dataset](https://github.com/asappresearch/abcd) (ABCD, MIT licence): about 10,000 role-played customer-support chats. Names, emails and addresses in it are synthetic.

- **Job A, triage:** broad intent (10 options), exact case (54 options), and a yes/no on product returns.
- **Job B, next action:** which of 30 playbook actions the agent should take next.

Every engine answers the same frozen 300 test chats per job (`data/samples/`). Question sets: `v2` is the company criteria only; `v3` adds the five most similar labelled chats from a pool of 500 (TF-IDF retrieval, see `src/bench/prompting.py`).

## Setup

```bash
uv sync --group gpu --group databricks --group dev   # torch (CUDA 12.8), peft, decider, databricks
cp .env.example .env                                  # JEV_API_KEY, OPENAI_API_KEY
mkdir -p data/raw/abcd && for f in abcd_v1.1.json.gz guidelines.json ontology.json; do
  curl -sSL -o data/raw/abcd/$f https://github.com/asappresearch/abcd/raw/master/data/$f; done
uv run pytest
```

For ai_decide and OpenJev, set `DATABRICKS_CONFIG_PROFILE` to a Databricks CLI profile with access to a serverless workspace.

## Running

```bash
uv run python -m bench.run --engine jev --dataset abcd --sample test300 --questions v2
#   --engine jev | openai_decisions | ai_decide | openjev | decider    --dataset abcd | actions    --questions v2 | v3
scripts/story.sh                      # all decider-2b GPU arms: calibrate, train, merge, score
scripts/seeds.sh; scripts/act5.sh     # seed reruns; shared adapter + adapter swap test
scripts/ai_decide_runs.sh             # ai_decide, both jobs, v2 and v3
scripts/openai_runs.sh                # OpenAI Decisions, both jobs, v2 and v3
uv run python scripts/ai_decide_check.py --n 150   # which model is behind ai_decide (agreement test)
uv run python -m bench.story_table    # rebuild the generated tables in RESULTS.md from results/runs.csv
```

Every call is appended to `results/runs.csv` (one row per call × question), which holds all the results behind RESULTS.md. Trained adapters are not included (about 900 MB); `scripts/story.sh` retrains them, about 35 minutes each on a 6 GB laptop GPU.

## Layout

| Path | What |
|---|---|
| `src/bench/data/` | `abcd.py` job A (triage), `abcd_actions.py` job B (next action); `*_v2.py` the company criteria |
| `src/bench/engines/` | One class per engine, all returning `{question: {label, score}}` |
| `src/bench/prompting.py` | The `v3` input: retrieved labelled examples added to each chat |
| `src/bench/run.py` | Runs an engine over a frozen sample and appends to `results/runs.csv` |
| `src/bench/lora/` | `train.py` (QLoRA), `merge.py` (into bf16), `calibrate.py` (refit temperatures on dev) |
| `src/bench/metrics.py`, `report.py`, `story_table.py` | Bootstrap intervals, paired McNemar tests, and the RESULTS.md tables |
| `data/samples/` | Frozen dev150 / test300 conversation ids per job |

Local GPU training needs a C compiler for Triton (Qwen 3.5's linear-attention kernels). If you have none, `pip install ziglang` and point `CC` at a wrapper that runs `python -m ziglang cc`.

## Licence

MIT, see [LICENSE](LICENSE). ABCD is MIT-licensed by ASAPP; decider-2b is Apache 2.0.
