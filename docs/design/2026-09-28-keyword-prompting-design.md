# Keyword prompting under imbalance: design

**Date:** 2026-09-28 · **Status:** draft for review · **Deadline:** core experiment done by 2026-10-04

**Working title (outcome-neutral):** *Whose Words Become Keywords? Keyword-Guided Prompting
for Faithful and Fair LLM Summarisation of Imbalanced Reviews*

The original project title was *Can Keyword-Based Prompting Improve Faithfulness and Reduce
Bias and Hallucination in Review Summarisation with Imbalanced Data?* The final paper title
is chosen after the results. If H1 holds: *Keyword Prompting Is a Double-Edged Sword:
Faithfulness, Majority Bias and Negation Hallucination in Imbalanced Review Summarisation*.

**No training anywhere.** Every model is used zero-shot, the keyword extractor is
unsupervised, and the NLI and sentiment scorers are used off the shelf.

This is a separate paper from FairSum H1. It lives in the same repo and reuses
`src/fairsum`. New code is added as a subpackage and new scripts. Existing
FairSum modules are left alone, apart from one additive keyword argument (§6).

---

## 1. Claim and positioning

REFER (Huang, Fayek & Zhang, NewSum 2025) makes opinion summaries fairer by
telling the model **how many** reviews hold each opinion. PAKPA (Tang et al.,
ACL 2024) makes review summaries more faithful by grounding them in **aspect
terms**, which are in effect keywords. Each paper leaves out the other's
concern:

| | REFER | PAKPA | SigExt (Xu et al. 2024) | This study |
|---|---|---|---|---|
| Signal given to the model | opinion counts | aspect terms + counts, per sentiment | salient keyphrases | keyphrases per sentiment group, with negation kept, ± counts |
| Faithfulness measured | no | human judgement only | AlignScore | NLI support, contradiction, count accuracy |
| Bias measured | yes (proportions) | no | no | proportions **and** content coverage per group |
| Imbalance | 50/50, 75/25 | natural | none | 50/50, 75/25, 87.5/12.5 in both directions |
| Output | paragraph | key-point list | paragraph | paragraph |
| Small models | fail (Llama-3-8B dropped) | 7B | 7B (Mistral became *less* faithful) | 1B–8B, compliance reported |

**Mechanism under test.** Salience-ranked keywords come mostly from the
majority opinion. Keyword prompting may therefore improve faithfulness while
amplifying majority bias. Negated keyphrases ("no side effects") can also be
reversed, which is a hallucination that is also a bias error. PAKPA reinforces
the concern from another side: its evaluation keeps only key points matched by
at least 15 comments, so prevalence ranking drops minority opinions by
construction.

## 2. Hypotheses (registered before any run)

Primary metrics are defined in §5. "Skewed" means the four non-balanced levels.

| | Hypothesis | Test | Falsified if |
|---|---|---|---|
| **H1** | `kw_global` vs `direct`: unsupported-proposition rate falls **and** majority amplification (MA) rises on skewed levels, more at 87.5/12.5 than at 75/25 | paired Wilcoxon per level; interaction condition × level | MA does not rise, or rises equally at both skews |
| **H2** | `kw_strat` vs `kw_global`: MA and coverage-parity gap fall, and the unsupported rate does not rise by more than 0.02 | paired Wilcoxon | MA or gap does not fall |
| **H3** | `kw_strat_neg` vs `kw_strat`: contradiction rate falls | paired Wilcoxon | no reduction |
| **H4** | `key_refer` vs `refer`: unsupported rate and count error fall, and \|Δp\| does not rise: the upper bound of the 95% CI on the mean paired difference stays below 0.05. On the 3B model, where REFER is expected to fail, `kw_strat_neg` beats `direct` on \|Δp\| | paired Wilcoxon; bootstrap CI for the non-inferiority parts | either part fails |

**Amendment (2026-09-29, before any score was seen; final review I2 and I4).**

- **H1.** H1a and H1b are registered on the four skewed levels **pooled**.
  Per-level paired effects are reported as a descriptive table and are not
  part of the Holm family. H1c's effect size is mean(Δ at 88 levels) −
  mean(Δ at 75 levels), with a bootstrap 95% CI.
- **H3.** H3 is tested only on collections where negation handling changed
  the `kw_strat_neg` prompt. On the others the two prompts are identical, so a
  pair there measures only batch noise. The share of such collections (the
  manipulation rate) is reported beside H3.

The H1 interaction is tested without a mixed model, which keeps statsmodels
out of the stack. Paired differences (`kw_global` − `direct`) at the 88 levels
are compared with those at the 75 levels by a one-sided Mann-Whitney U. H2's
"does not rise by more than 0.02" is judged the same way as H4's
non-inferiority: the CI upper bound must stay below 0.02.

**Pre-generation evidence for H1:** the *keyword minority share*, the share of
`kw_global` phrases found in at least one minority review, compared with the
minority share of reviews. It is computed in notebook 1 before any model runs.

All primary comparisons use Holm correction. Effect sizes (median paired
difference with a bootstrap 95% CI) are reported beside every p-value. A null
result is reported as a null.

## 3. Data

Both sources are downloaded as raw JSONL with `huggingface_hub`. `datasets` 4.5
no longer runs loading scripts, so the Amazon repo's script cannot be used.

| Source | File | Entity | Polarity from | Why |
|---|---|---|---|---|
| **Amazon Reviews 2023** (REFER's source) | `McAuley-Lab/Amazon-Reviews-2023` `raw/review_categories/Health_and_Personal_Care.jsonl` (227 MB) | `parent_asin` | stars: 4–5 positive, 1–2 negative, 3 dropped | direct comparison with REFER; health-adjacent products |
| **Drugs.com** (UCI 462) | `lewtun/drug-reviews` `train.jsonl` + `test.jsonl` | `drugName` | rating 1–10: ≥ 7 positive, ≤ 4 negative | medical stakes; many reviews per drug |

**Review filter:** 30–120 words, as in REFER. HTML entities are unescaped.
Exact duplicate texts are removed.

**Collections.** 8 reviews of one entity, split by polarity into five levels
(pos:neg):

| level | `bal` | `pos75` | `pos88` | `neg75` | `neg88` |
|---|---|---|---|---|---|
| pos:neg | 4:4 | 6:2 | 7:1 | 2:6 | 1:7 |

REFER used the first three kinds of level (balanced, 75/25, 25/75). The two 88
levels are the extension. Per dataset and level, there are **60 test and 10 dev
collections**, 700 in total. Entities are sampled uniformly at random under
seed 42, never alphabetically (lesson F4). An entity is used at most once per
level, and no review appears twice in a collection. Dev and test entities are
disjoint. Review order within a collection is shuffled under the seed, so the
minority is not always last. The result is written once to
`data/kw/collections.jsonl` with a manifest (`data/kw/manifest.json`).

**Why star-rating polarity is legitimate.** Review platforms expose ratings, so
per-review polarity is available at inference time. Stratified keywords and
REFER's oracle counts use the **same** information, which makes that comparison
fair. A label-free variant (polarity from a sentiment classifier) is a stretch
ablation.

## 4. Conditions

Every condition wraps REFER's direct prompt verbatim: *"Reviews about {topic}.
Each review is separated by ||: {source}. Please write a short text containing
the salient information (i.e., a summary). The summary of the reviews is:"*.
`{topic}` is the drug name (Drugs.com) or, for Amazon, the product title from
the category's `raw/meta_categories/meta_<Category>.jsonl`. If a title is
missing, it falls back to "a Health and Personal Care product".

| id | Family | What is added |
|---|---|---|
| `direct` | baseline | nothing |
| `refer` | REFER | "Let's first determine how many reviews out of 8 are positive, negative. Then, let's generate a balanced summary that accurately reflects the frequency of these opinions." |
| `cot_refer` | REFER | "Let's think step by step." + `refer` (REFER's strongest variant) |
| `refer_oracle` | REFER upper bound | "{p} and {n} out of 8 are positive, negative, generate a balanced summary reflecting this distribution." |
| `kw_global` | keywords | "Consider including the following keywords: k1, …, k10" (SigExt wording), keywords extracted from all 8 reviews |
| `kw_strat` | keywords | "Positive reviews mention: …; negative reviews mention: …", 5 keywords per group (equal quota) |
| `kw_strat_neg` | keywords (proposed) | as `kw_strat`, with negation kept in keyphrases (§4.1) |
| `key_refer` | combined (proposed) | `refer` + `kw_strat_neg` |
| `key_oracle` | combined upper bound | `refer_oracle` + `kw_strat_neg` |

`kw_global` → `kw_strat` isolates stratification, and `kw_strat` →
`kw_strat_neg` isolates negation handling. The extractor is the same
throughout.

### 4.1 Keyword extraction (KeyBERT-style, implemented locally)

1. **Candidates:** adjective-noun phrases of 1–3 words, following PatternRank
   (Schopf et al. 2022): part-of-speech pattern `ADJ* (NOUN|PROPN)+`, tagged
   with spaCy `en_core_web_sm`. A phrase may not start or end on a stop word
   or contain a negation cue. *(Changed 2026-09-29 after notebook 1 showed
   that plain n-grams yield fragments such as "reviews talked".)*
2. **Ranking:** cosine similarity between each candidate and the text it
   summarises, embedded with `sentence-transformers/all-MiniLM-L6-v2`. The text
   is all reviews (`kw_global`) or one polarity group (`kw_strat*`). Candidates
   are then chosen by MMR (diversity 0.5), skipping any phrase whose words are
   a subset or superset of an already chosen one.
3. **Negation kept (`kw_strat_neg` only):** the **same phrases** as `kw_strat`,
   so selection is identical and only the surface form differs. For each
   phrase, its occurrences in its own group's reviews are checked for a
   negation cue (`no, not, never, nothing, none, without, hardly, cannot, nor`
   and negative contractions such as `didn't`) in the preceding 3 tokens. If
   most occurrences are negated, the phrase is replaced by its most frequent
   literal source span, starting at the cue ("no side effects", "didn't have
   any side effects").
4. **k = 10** in total (5 per group when stratified). A group with fewer usable
   candidates gives its unused quota to the other group. Keyword lists are
   cached per collection, so every model sees identical keywords.

**Stretch (only if the core finishes early):** `kw_aspect`, a PAKPA-lite
condition built from noun-chunk aspects plus sentence sentiment, with no LLM
aspect analysis.

## 5. Metrics

**Propositions.** Summaries are split into sentences, then split again at
contrast markers (`but`, `however`, `although`, `while`, `whereas`, `;`). Count
statements ("6 of the 8 reviews are positive") are removed from the proposition
set and scored separately. So are procedural or meta lines ("Step 1: …",
"Let's …", "Here is a summary"): `cot_refer` emits its reasoning inline, and
scoring those lines would inflate its unsupported rate. The number removed is
recorded per output. REFER split summaries with GPT-4o-mini, so this is a
documented deviation; an LLM splitter can replace it when API keys arrive.

**NLI matrix.** `microsoft/deberta-large-mnli` (as in `verifier.py`) scores
every (review → proposition) pair, with the review as premise. It yields
P(entail) and P(contradict) per pair. Most faithfulness and content-fairness
metrics come from this one matrix.

| Group | Metric | Definition | Primary |
|---|---|---|---|
| Faithfulness | **unsupported rate** | share of propositions with max over reviews of P(entail) < 0.5 | ✔ |
| Faithfulness | support score | mean over propositions of max P(entail) (SummaC-ZS-like) | |
| Hallucination | **contradiction rate** | share of propositions with max P(contradict) ≥ 0.5 and max P(entail) < 0.5 | ✔ |
| Hallucination | **count error** | \|stated − true\| for parsed "X out of 8 …" statements; also the share of outputs with a wrong count | ✔ (REFER family) |
| Hallucination | invented numbers | numbers in the summary absent from the source (count statements excluded) | |
| Fairness (proportion) | **\|Δp\|** | \|p_neg(summary) − p_neg(source)\|, where p_neg(summary) is the share of polar propositions a 3-class sentiment classifier labels negative | ✔ |
| Fairness (proportion) | **MA** (majority amplification) | p_maj(summary) − p_maj(source), skewed levels only; > 0 means the majority is amplified | ✔ |
| Fairness (proportion) | BUR, UER, SOF | `fairsum.metrics.fairsum_metrics(labels=["positive","negative"])`, for REFER comparability. With a binary attribute, UER = SOF = \|Δp\|/2 (vault F2), and this is stated in the paper | |
| Fairness (content) | **coverage-parity gap** | R_maj − R_min, where R_g is the share of group-g reviews that entail at least one proposition | ✔ |
| Fairness (content) | attribution share | share of supported propositions backed by at least one minority review, compared with the minority share | |
| Manipulation check | keyword uptake | share of prompt keywords that appear in the summary (lemmatised, fuzzy) | |
| Quality | length, compliance | words; status ∈ {ok, truncated, empty, no_summary, refusal} | reported beside everything |

**Sentiment classifier (as run).** A hybrid instrument, replacing the
original 3-class plan after the pilot:

- `cardiffnlp/twitter-roberta-base-sentiment-latest` decides only **neutral vs
  opinion**.
- `siebert/sentiment-roberta-large-english` decides **polarity** for
  opinions.

The reasons:

- cardiffnlp alone failed the 0.90 gate (0.76 against star polarity).
- siebert passes (0.91) but is at least 0.98 confident even on neutral
  facts, so a confidence band cannot find neutrality.

Neutral propositions are excluded from p_neg. The hybrid is validated on
source reviews with two numbers: polarity accuracy on the non-neutral calls,
and the neutral rate by true polarity (a gate that neutralises one side more
than the other would bias |Δp|). A human check of about 100 summary
propositions is required before submission.

**Human check protocol (registered 2026-09-29, before any score was computed).**

- **Sample.** 25 summaries, 4 propositions each (100 items). Summaries are
  drawn with seed 42 from scored test generations (status `ok`, at least 4
  propositions), evenly over model × dataset and spread over conditions.
  Propositions are drawn at random within each summary. The sample is drawn
  after every generation run has finished.
- **Blinding.** Annotators see the 8 reviews and the full summary with the
  target sentences numbered. They never see the model, condition, level,
  review ratings or scorer labels. The item-to-generation key is stored
  outside the annotation folder.
- **Labels per item.** *claim*: yes / no (no = heading, meta line, bare count
  or fragment; the other two labels are then left blank). *sentiment*:
  positive / negative / neutral / mixed, judged on the sentence alone.
  *support*: supported (at least one review states or clearly implies it;
  quantity words such as "many" or "most" are ignored) / contradicted (no
  review supports it and at least one says the opposite) / unsupported (no
  review supports or contradicts it). "Supported wins" matches the scorer.
  A sentence about the balance of opinion ("received mixed reviews") is
  judged against the 8 reviews as a whole. The exact wording is the guide in
  `src/fairsum/kwsum/human_check.py`, shown to annotators in the sheet and the
  reading pack.
- **Reported.** For each label: scorer–human accuracy, Krippendorff's α
  (nominal) and the confusion matrix. Sentiment is compared on claim = yes,
  non-mixed items, and the mixed rate is reported. The human and scorer
  unsupported rates are compared, with a bootstrap CI resampling summaries.
  With a second annotator, human–human α is reported too.
- **Interpretation, fixed in advance.** α ≥ 0.80 is good and 0.667–0.80 is
  acceptable (Krippendorff's thresholds). Below 0.667 on sentiment, the
  proportion metrics (|Δp|, MA) become exploratory. Below 0.667 on support, the
  unsupported and contradiction rates become exploratory. A scorer that is
  systematically stricter than humans is reported as giving upper bounds.

**Count and meta handling (as run, final review C1/C2).**

- A clause that states a count *and* an opinion ("Two reviewers reported
  nausea") is kept with a generic subject ("Some reviewers …").
- Only pure count statements are removed and parsed for count accuracy.
- Counts larger than the collection are ignored.
- Text before a final "Summary:" header, per-review notes, instruction
  echoes and title-case headings are meta.
- A truncated output loses its unterminated last sentence.

**Failure policy** (the FairSum invariant). A non-compliant output gets a
status and no metric values. It is never scored as 0 or as maximally unfair.
Compliance rate is reported for every (model, condition). A sensitivity table
treats failures as worst case.

**Deferred until keys arrive:** a judge from a different family (G-Eval-style
faithfulness and balance ratings) on a stratified subsample. MiniCheck or
AlignScore will be added only if they load under transformers 5.2. The pilot
checks this; the NLI metrics above do not depend on it.

## 6. Models and decoding

| Model | Precision | Why |
|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | bf16 | smallest rung of the Llama size ladder (added, see below) |
| `meta-llama/Llama-3.2-3B-Instruct` | fp16 if it fits at batch 4, else NF4 | the small end, where REFER is expected to fail |
| `meta-llama/Llama-3.1-8B-Instruct` | NF4 | REFER's small model (direct replication point) |
| `Qwen/Qwen2.5-7B-Instruct` | NF4 | second family |
| `mistralai/Mistral-7B-Instruct-v0.3` | NF4 | the model SigExt found keywords hurt (H3 test) |

**Amendment (2026-09-29, before any score was computed).**

- **Scope: small open models only.** The registered comparisons use the five
  local models above (1B–8B). API models are an optional reference arm, not
  part of the registered hypotheses. The paper's claim is about small models
  that can run on one consumer GPU, where REFER reports failure.
- **Llama-3.1-8B** runs from `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, a
  pre-quantised NF4 copy of the same weights: the Meta repo is gated for this
  account (403). This follows the §8 rule for gated models, with no change of
  model.
- **Llama-3.2-1B** is added so that one family gives a size ladder (1B / 3B /
  8B). It was already cached locally and costs about two GPU hours.
- The model set is now fixed. Every model is reported whatever its result.

Decoding follows REFER: greedy (their temperature 0.001 is effectively greedy),
`max_new_tokens=256`, `repetition_penalty=1.1`, and the chat template with the
prompt as the user turn. The whole completion is the summary; REFER's prompts
use no marker. `BatchedGenerator.generate` gains an optional
`repetition_penalty=None` argument. This is additive and existing calls are
unchanged.

**API backends.** An OpenAI-compatible client (OpenAI, OpenRouter and Groq via
`base_url`) and an Anthropic client share one `Backend` interface with the local
backend. Keys are read from environment variables only and never written to
disk or logs. They are built now and exercised once keys are provided.

## 7. Architecture

```
src/fairsum/kwsum/
  sources.py       download and stream-filter raw JSONL → review pool (entity, text, rating, polarity)
  collections.py   imbalanced 8-review collections, dev/test by entity, seeded, manifest
  keywords.py      KeyBERT-style extractor; global / stratified / negation-kept; cache
  prompts.py       the 9 condition builders (REFER text verbatim); condition registry
  backends.py      Backend protocol; HFBackend (wraps BatchedGenerator); OpenAICompatBackend; AnthropicBackend
  compliance.py    status classification of raw outputs
  propositions.py  sentence and contrast splitting; count-statement parser
  scoring.py       NLI matrix, sentiment, all §5 metrics → one row per generation
  stats.py         paired tests, Holm, bootstrap CIs, level-interaction tables
  config.py        TOML run config → dataclasses
configs/kw_week1.toml                 datasets, levels, N, models, conditions, decoding
scripts/kw_build_collections.py       sources → collections + keywords (CPU, minutes)
scripts/kw_generate.py                resumable; appends JSONL keyed by (collection, condition, model)
scripts/kw_score.py                   generations → scores.csv (GPU, resumable)
scripts/kw_analyze.py                 scores → tables/*.csv + figures/*.png
notebooks/kw_01_data_and_keywords.ipynb  explore pools, imbalance, keyword examples, extractor sanity checks
notebooks/kw_02_pilot.ipynb              small end-to-end run, raw output inspection, throughput, classifier check
notebooks/kw_03_results.ipynb            hypothesis tests, figures, example summaries for the paper
tests/test_kw_*.py                       pure-logic tests (collections, keywords, prompts, compliance, propositions, metrics, stats)
```

**Layout choice.** Notebooks are thin and import from the package; overnight
runs go through the scripts. Everything under `results/kw/<run_id>/` is
append-only JSONL or CSV, so an interrupted run resumes where it stopped.

## 8. Compute plan and decision rules

- **Test matrix per model:** 600 collections × 9 conditions = 5,400
  generations. Dev adds 900.
- **Estimate (unmeasured for 7–8B NF4):** about 0.4–0.6 generations/s, or 3–4 h
  per model, so two overnight runs cover four models.
- **Pilot (Day 2):** 1 dataset × 5 levels × 4 collections × 9 conditions on the
  3B model (180 generations), plus 16 generations on each 7–8B model for
  throughput and VRAM.

**Decision rules after the pilot:**

- If 7–8B runs below 0.25 generations/s, cut test N to 40 per cell before
  cutting a model.
- If a model's compliance is below 50% on a condition, keep it and report it:
  that is a finding (REFER's 8B failure; our F5).
- If any gated model is inaccessible, substitute the same family's next open
  model and note it.

## 9. Schedule (Mon 28 Sep → Sun 4 Oct)

| Day | Work |
|---|---|
| Mon | spec, plan; sources, collections, keywords, prompts (with tests) |
| Tue | backends, compliance, generation script; pilot; propositions and scoring; **overnight:** 3B + 8B |
| Wed | scoring on the pilot, then on the overnight output; stats; **overnight:** Qwen 7B + Mistral 7B |
| Thu | score everything; analysis tables and figures |
| Fri | results notebook; findings note in the vault; API backends ready |
| Sat–Sun | buffer; API models if keys have arrived |

## 10. Out of scope this week

Human evaluation of summary quality (the ~100-proposition check of the
scorers in §5 is in scope); the reference-summary datasets (SPACE, Yelp, AmaSum); the
CADEC and PsyTAR gold adverse-event datasets; multiple samples per input (the
FairSum H1 link); PAKPA-style LLM aspect extraction. Each is listed as a
journal-version extension.

## 11. Known deviations from REFER and risks

**Deviations:**

- Product category chosen by us (REFER does not state theirs).
- Rule-based proposition splitting instead of GPT-4o-mini.
- Proportions estimated by a classifier, not by BARTScore.
- N = 60 instead of 300 per level.

**Risks:**

- Gated models (Llama, Mistral) need the licence accepted on the logged-in
  Hugging Face account.
- NF4 on transformers 5.2 has not been exercised for 7–8B in this repo.
- Keyword uptake may be low for small models, which would make H1 and H2
  untestable for them. The uptake manipulation check exists to catch this.
