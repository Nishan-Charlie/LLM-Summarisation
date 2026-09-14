# fairsum

Risk-controlled fair summarization of patient drug reviews.

A conformal procedure that certifies, distribution-free, how much reported
adverse-event signal a summary may drop — and abstains when it cannot meet the
bound. Test-time compute is the actuator; the uncertainty of the decode is the
controller.

Design notes live in the Obsidian vault at
`Research Vault/Research Vault/Prathyuman-LLM Summarisation`.

## Install

```bash
pip install -e ".[dev]"          # pure-logic modules + tests
pip install -e ".[dev,annotate]" # adds openpyxl for the annotation sheet
pip install -e ".[dev,gen]"      # adds torch / transformers for generation
pytest                           # 280 tests, ~2s
```

Verified on Python 3.12.3, torch 2.6.0+cu124, transformers 5.2.0,
RTX 4070 Laptop (8 GB).

## Week 1

| Day | Command | Gate |
|---|---|---|
| 1 | `python scripts/run_audit.py --csv <v1_results>.csv` | is v1 salvage or scrap? |
| 2 | `python scripts/bench_throughput.py --model meta-llama/Llama-3.2-1B-Instruct` | what `k` is affordable? |
| 3 | `python scripts/build_pool.py --seed 42 --annotation-batch 50` | annotate 50, time it honestly |
| 4 | `python scripts/check_agreement.py --a a.jsonl --b b.jsonl` | α ≥ 0.7 freeze · < 0.6 stop |
| 5 | `python scripts/power_analysis.py --alpha 0.1` | is effective *n* instances or drugs? |

`run_audit.py` works with no arguments — it prints the degenerate-row fraction
predicted from v1's published aggregates alone. Pass `--csv` to measure the real
rate and settle it.

## Modules

| Module | Purpose |
|---|---|
| `metrics` | FairSumm metrics (Zhang et al. Defs 5.1–5.4), degenerate counts **reported not scored** |
| `audit` | the Day 1 diagnostic and the back-out estimator |
| `data` | Druglib loading (direct-zip; no `ucimlrepo` dependency) |
| `instances` | instance construction, drug-disjoint splits, AE-density conditions |
| `annotation` | AE schema, JSONL storage, Krippendorff's α |
| `prompts` | six prompt strategies, repair prompt, `FINAL SUMMARY:` parsing |
| `generation` | batched k-sample decoding with full token accounting |
| `verifier` | NLI coverage checking; scoring and measurement kept separate |
| `conformal` | CRC threshold fitting, risk–coverage curves |

## Three invariants the code enforces

**A failure is never a score.** v1's judge returned all-zero counts on a JSON
parse error, which became SPD=UER=0.2, BUR=1, AUC=1.0 — indistinguishable from a
genuinely unfair summary. Here `normalize_counts` raises, `Metrics` carries a
`status`, the verifier has an `ABSTAIN` verdict, and `parse_final_summary`
returns `NO_MARKER` rather than falling back to raw text.

**The judge is never the generator.** `verifier.py` takes an injected entailment
model and the docstring says why. Scoring (label-free, drives selection) and
measurement (gold labels, computes the loss) are separate entry points.

**Rejected work still counts.** Every generation is recorded with its token
counts, including samples that lose selection, because H2 compares parallel
against sequential scaling per FLOP.

## Measured on this machine (RTX 4070 Laptop, 8 GB)

Llama-3.2-1B fp16, 8-review prompts, `max_new_tokens=320`:

| method | batch | gen/s | peak VRAM | 8,000 gens | summary delivered |
|---|---:|---:|---:|---:|---:|
| `direct` | 8 | 0.76 | 5.3 GB | 2.9 h | **100%** |
| `ssfp` | 8 | 0.34 | 6.0 GB | 6.5 h | **58%** |

**Cap the batch at 8.** Batch 16 peaked at 9.37 GB — above physical VRAM — and
Windows spilled to host memory rather than OOM-ing, collapsing throughput to
0.24 gen/s, *worse* than batch 8.

## Three findings the code produced

**The v1 degenerate-row estimate.** From the published `method_table` alone,
using `SPD ≡ 2·UER` for genuine distributions: ~26% of rows overall, ranging
6.7% (`ssfp`) to 47.8% (`direct`). A 7× spread across methods means the reported
ranking may track judge failure rather than summary fairness. `run_audit.py`
tests it against the real rows.

**Grouping by indication costs 70% of the pool.** Druglib's `condition` is free
text averaging 4.5 distinct indications per drug (max 30). Grouping on
(drug, indication) leaves 42 groups with ≥8 reviews; grouping by drug leaves 144.

So the default is `group_by="drug"` with `condition_name=None`, and prompts then
make *no* indication claim. v1's bug was asserting an indication the reviews were
not about — a prompt that claims nothing cannot claim something false.
`group_by="drug_indication"` remains available for a stricter ablation.

**Reasoning prompts don't deliver a summary 42% of the time at 1B.** `direct`
delivers 100%; `ssfp` delivers 58%. Inspecting raw output shows the failures are
*not* truncation — one sample ran 66 tokens of the 320 budget, finished its
enumeration, and emitted EOS having never written a summary.

So a prompt strategy that fails to produce a summary 42% of the time cannot be
fairly compared against one that always does. v1's ranking of `direct` against
`ssfp`/`ecvp` was partly measuring compliance. Vault note
`F5 Reasoning Prompts Do Not Deliver`.

The marker requirement is now a per-prompt contract (`PROMPT_SPECS`) rather than
a blanket rule: prompts with no reasoning scaffold need no delimiter, since the
whole completion is the summary. That alone took `direct` from 44% to 100% —
the prompt used to *end* on the marker, so the model continued straight into the
summary without echoing it and good output was discarded.

## Status

Week 1 scaffolding, tested, and every code path executed at least once —
including the GPU generation and NLI verifier paths, via `scripts/smoke_e2e.py`.

Nothing here has produced a research result. `smoke_e2e.py` uses **placeholder
keyword-matched events, not gold annotations**, and prints numbers that are
meaningless by construction; it exists to prove the plumbing connects. No v1
number should be cited until `run_audit.py` runs against the real CSV.
