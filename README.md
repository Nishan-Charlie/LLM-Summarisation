# Keeping Minority Opinions in Review Summaries: Do Keyword Prompts Help Small Language Models?

Code, data splits and result tables for a study of keyword-guided prompting when the reviews being
summarised are imbalanced, for example seven positive reviews and one negative.

Keyword extractors rank phrases by salience, so under imbalance most keywords come from the majority
opinion. The study tests what that does to a summary: whether keywords make it more faithful, whether
they push the minority opinion out, and whether grouping keywords by opinion brings it back.

## What was run

| | |
|---|---|
| **Inputs** | 700 collections of 8 reviews about one product or drug (600 test, 100 dev), from Amazon Reviews 2023 (Health and Personal Care) and Drugs.com |
| **Imbalance levels** | positive:negative of 4:4, 6:2, 7:1, 2:6 and 1:7 (`bal`, `pos75`, `pos88`, `neg75`, `neg88`) |
| **Prompts** | 9 registered conditions and 3 control prompts (table below) |
| **Models** | Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B and Mistral-7B, all instruction-tuned, zero-shot, greedy decoding |
| **Scale** | 600 test collections × 9 conditions × 5 models = 27,000 test summaries in the main run |
| **Scoring** | NLI for faithfulness and coverage, and a sentiment classifier for opinion proportions |

The hypotheses and primary metrics were registered before any model was run. Later changes are
recorded as dated amendments. Both are in [docs/design/](docs/design/).

## Main findings

- Grouping keyphrases by positive and negative opinion helped every model cover more of the minority
  view, and not only because the summaries became longer.
- That gain costs proportionality. Giving both sides equal space in the prompt pushes the summary
  toward an even split, so the minority looks more common than it is.
- The effect of keywords on unsupported statements depended on the model.

These findings come from the automatic scorers. The human evaluation described below is not
finished. The numbers are in [results/kw/week1/tables/](results/kw/week1/tables/) and
[results/kw/controls/tables/](results/kw/controls/tables/).

## Repository layout

```
configs/                  run configurations: kw_week1.toml (main run), kw_controls.toml (control prompts)
data/kw/                  collections.jsonl, keywords.jsonl and the manifest that pins them
docs/design/              registered design and its dated amendments
src/fairsum/kwsum/        the study's code: pools, collections, keywords, prompts, backends, scoring, statistics
src/fairsum/              shared helpers the study imports: batched generation, fairness metrics, agreement statistics
scripts/                  command-line entry points, one per pipeline step (kw_*.py) and a Windows run queue
notebooks/                data and keyword exploration, pilot run, results
results/kw/week1/         tables and figures of the main run
results/kw/controls/      tables of the control-prompt run
annotations/kw_human_check/   annotation sheets and the blinded reading pack
tools/kw_label_site/      web page used for labelling
tests/                    unit tests
```

## Setup

Python 3.10 or newer. The results were produced with Python 3.12, PyTorch 2.6 and transformers 5.2
on a single consumer GPU with 8 GB of memory.

```bash
pip install -e ".[gen,kw,annotate,dev]"
python -m spacy download en_core_web_sm
```

| Extra | Needed for |
|---|---|
| `gen` | local generation and scoring (PyTorch, transformers, bitsandbytes) |
| `kw` | keyword extraction, statistics, figures, dataset download |
| `annotate` | reading and writing the annotation sheets |
| `dev` | running the tests |
| `api` | optional OpenAI-compatible and Anthropic backends, which the registered runs do not use |

Model weights come from the Hugging Face Hub. The two Llama-3.2 checkpoints are gated, so request
access on their model pages and log in with `huggingface-cli login`. The 7–8B models load as
pre-quantised 4-bit checkpoints and need a CUDA GPU.

Check the install:

```bash
pytest
```

## Reproducing the results

Every step reads one config file. Generation and scoring append to JSONL files as they go, so an
interrupted run continues where it stopped when you start it again.

**1. Collections and keywords** (optional; CPU)

`data/kw/collections.jsonl` and `data/kw/keywords.jsonl` are already in the repository, so you can
skip this step. To rebuild them from the source datasets:

```bash
python scripts/kw_build_collections.py --config configs/kw_week1.toml
```

This downloads the raw reviews from the Hub, filters them into review pools (cached under
`data/kw/pools/`), samples the collections under seed 42 and extracts the keywords.

**2. Generate summaries** (GPU; once per model)

```bash
python scripts/kw_generate.py --config configs/kw_week1.toml --model llama3.2-3b
```

Model names are `llama3.2-1b`, `llama3.2-3b`, `llama3.1-8b`, `qwen2.5-7b` and `mistral-7b`. For a
quick pilot, add `--datasets amazon --split test --limit-per-cell 4`. Use `--dry-run` to see what
would be generated.

**3. Score** (GPU)

```bash
python scripts/kw_score.py --config configs/kw_week1.toml
python scripts/kw_score.py --config configs/kw_week1.toml --validate-sentiment
```

The second command checks the sentiment classifier against the reviews' star ratings.

**4. Control prompts**

The controls reuse the same collections, keywords, decoding and scorer. `direct_len` matches the
length of the main run's `kw_strat` summary for the same collection, so run it after step 2.

```bash
python scripts/kw_generate.py --config configs/kw_controls.toml --model llama3.2-3b
python scripts/kw_score.py --config configs/kw_controls.toml
```

**5. Tables and figures**

```bash
python scripts/kw_analyze.py --config configs/kw_week1.toml
python scripts/kw_amendment_analysis.py --config configs/kw_week1.toml --controls configs/kw_controls.toml
python scripts/kw_export_results.py --config configs/kw_week1.toml
```

| Script | Writes |
|---|---|
| `kw_analyze.py` | the registered tests, condition means, compliance and figures, under `results/kw/week1/` |
| `kw_amendment_analysis.py` | entity-clustered tests, majority amplification by direction, and the control-prompt tests |
| `kw_export_results.py` | `all_results.csv`, every result in one long table, and `all_condition_means_wide.csv` |
| `kw_paper_assets.py` | the LaTeX numbers, tables and figures for the manuscript, into `paper/` (the manuscript is not in this repository) |

On Windows, `scripts/kw_run_queue.ps1` runs generate, score and analyse for a list of models
unattended. Its `-BigModelCache` parameter defaults to a path on the authors' machine, so set it
to your own Hugging Face cache.

### What is and is not in `results/`

The tables and figures are included. The raw generations and per-summary scores
(`generations.jsonl`, `scores.jsonl`) are not, because of their size; steps 2 to 4 recreate them.
Each run folder has a `run_manifest.json` recording the inputs it was produced from. A run refuses
to continue if the collections or keywords have changed since it started.

## Prompt conditions

Every condition wraps the same base prompt: the reviews, followed by a request for a short summary.

| Condition | What is added to the base prompt |
|---|---|
| `direct` | nothing |
| `refer` | an instruction to count the positive and negative reviews first, then write a balanced summary reflecting those frequencies |
| `cot_refer` | "Let's think step by step." plus `refer` |
| `refer_oracle` | the true counts, with the instruction to write a balanced summary reflecting them |
| `kw_global` | 10 keyphrases extracted from all 8 reviews |
| `kw_strat` | 5 keyphrases from the positive reviews and 5 from the negative ones, listed by group |
| `kw_strat_neg` | as `kw_strat`, with negation kept in the phrase ("no side effects") |
| `key_refer` | `refer` plus `kw_strat_neg` |
| `key_oracle` | `refer_oracle` plus `kw_strat_neg` |

Controls, added by amendments 9 and 10:

| Condition | Purpose |
|---|---|
| `direct_len` | the plain prompt with a target length equal to the `kw_strat` summary's, to separate the keyword effect from a length effect |
| `counts_neutral` | the true counts with no instruction about balance |
| `counts_reflect` | the true counts with "generate a summary reflecting this distribution", without the word "balanced" |

The `refer` prompts reproduce the wording of REFER (Huang, Fayek and Zhang, NewSum 2025), and
`kw_global` uses the keyword wording of SigExt (Xu et al., 2024). Keyphrases are adjective–noun
phrases of one to three words, ranked by embedding similarity to the text they come from and chosen
by maximal marginal relevance ([src/fairsum/kwsum/keywords.py](src/fairsum/kwsum/keywords.py)).

## Metrics

A summary is split into propositions. Each proposition is checked against each of the 8 reviews
with an NLI model (`microsoft/deberta-large-mnli`) and labelled by a sentiment classifier.

| Metric | Meaning |
|---|---|
| Unsupported rate | share of propositions that no review entails |
| Contradiction rate | share of propositions that a review contradicts and none entails |
| Count error | difference between the review counts a summary states and the true counts |
| \|Δp\| | gap between the share of negative opinion in the summary and in the reviews |
| Majority amplification (MA) | majority share in the summary minus majority share in the reviews; above 0 means the majority is amplified |
| Coverage-parity gap | share of majority reviews reflected in the summary minus share of minority reviews reflected |

An output that is empty, a refusal, or contains no summary gets a status and no metric values. It is
never scored as zero. Compliance is reported for every model and condition, and one table treats
failures as the worst case. The registered tests are also reported with reviewer attribution removed
before NLI ("Some reviewers found…") and with entities as the statistical unit. Full definitions are
in section 5 of the [design document](docs/design/2026-09-28-keyword-prompting-design.md).

## Human evaluation

The protocol checks the automatic scorers against human labels on a blinded sample of 100 summary
statements. Annotators see the reviews and the summary, and never the model, prompt condition,
imbalance level or scorer output.

```bash
python scripts/kw_human_check.py make  --config configs/kw_week1.toml   # draw the blinded sample
python scripts/kw_human_check.py check --config configs/kw_week1.toml   # compare labels with the scorer
```

Annotators can fill the spreadsheets in `annotations/kw_human_check/` or label on the web page in
[tools/kw_label_site/](tools/kw_label_site/), a static page with a small serverless API. The
`web-items` and `from-web` actions of `kw_human_check.py` move items to the page and labels back.

The spreadsheets in the repository hold the blinded sample with empty label columns. Completed
labels are not included yet.

## Design record

| Document | Date | Content |
|---|---|---|
| [Design](docs/design/2026-09-28-keyword-prompting-design.md) | 2026-09-28 | claim, hypotheses H1–H4, data, conditions, metrics, models, decision rules |
| [Amendment 9](docs/design/2026-10-02-kw-amendment-9.md) | 2026-10-02 | two control prompts, entity-clustered inference, whole-summary human evaluation |
| [Amendment 10](docs/design/2026-10-05-kw-amendment-10.md) | 2026-10-05 | a third control prompt, an absolute-distortion check, human-evaluation fixes |

Earlier amendments are recorded inside the design document. Each amendment states what was already
known when it was written and what had not yet been generated or labelled.

## Data sources

| Dataset | Source on the Hugging Face Hub |
|---|---|
| Amazon Reviews 2023, Health and Personal Care | `McAuley-Lab/Amazon-Reviews-2023` |
| Drugs.com reviews (UCI Drug Review dataset) | `lewtun/drug-reviews` |

Reviews are kept if they have 30 to 120 words. Polarity comes from the rating: 4–5 stars positive
and 1–2 negative for Amazon, 7 and above positive and 4 and below negative for Drugs.com. Use of the
review texts in `data/kw/` is subject to the terms of the source datasets.

`scripts/kw_download_external.py` fetches several other review datasets into `data/external/`. The
runs reported here do not use them.
