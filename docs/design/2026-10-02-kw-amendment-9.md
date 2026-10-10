# Amendment 9: controls, clustered inference, whole-summary human evaluation

Keyword-prompting study (spec: `2026-09-28-keyword-prompting-design.md`).
Fixed on 2026-10-02, after all 27,000 week-1 summaries were scored, but **before any control
summary was generated and before any human label existed** (both labelling routes showed
0 of 100 labelled when this was written).

Prompted by an external review of the draft. The review's points, and what this amendment does
about each, are listed at the end.

## 1. Scope and primary claim

- The paper is the keyword study only. FairSum (repeated decoding, adverse-event omission,
  repair, conformal risk control) is a separate paper. It is not mentioned in this one: no
  results, methods, future-work sentence or citation. This is the authors' decision of
  2026-10-03.
- **Primary question:** when small models summarise imbalanced reviews, do keyword prompts raise
  minority coverage without distorting opinion proportions or adding unsupported claims?
- This is registered H2:
  - H2b is the primary outcome: coverage gap, `kw_strat` vs `kw_global`.
  - H2a (majority amplification) is the proportion guardrail.
  - H2c (unsupported rate, margin 0.02) is the faithfulness guardrail.
  - H1, H3 and H4 stay as registered and are reported as secondary.
- **Comparison family for H1–H4:** as in the original spec and code, Holm correction runs across
  the registered tests within each model × dataset. A hypothesis is reported as the number of the
  10 model × dataset cells in which it is supported.
- *Correction, 2026-10-03:* the first version of this line said the registered family was "each
  hypothesis across its 10 cells". That misdescribed the analysis registered on 2026-09-28, which
  this amendment does not change.
- **Comparison family for the new control tests (§2):** as written here on 2026-10-02, Holm runs
  across the 10 cells of each control hypothesis. It is kept as written even though the per-cell
  family would have supported C2 in one more cell (7 instead of 6 of 10).
- **False balance is a conditional, secondary finding.** It is claimed for a model, prompt and
  majority direction only if both hold:
  - (i) mean MA lies beyond the misclassification envelope for that level (−0.036 at 6:2/2:6,
    −0.054 at 7:1/1:7, at the measured 7.2% polar error);
  - (ii) the human perceived-split judgments (part B, Q2) go the same way.
  - Otherwise it is reported as within instrument error.

## 2. Two control prompts

Both run on all 600 test collections × 5 models (6,000 summaries). Decoding is the same as week 1.
They are stored as a separate run (`results/kw/controls/`), so the registered week-1 tables do
not change.

| Condition | Prompt | Isolates |
|---|---|---|
| `direct_len` | The plain prompt, plus "Write about N words." in the slot keyword guidance uses. N is the word count of the same model's `kw_strat` summary for the same collection, rounded to the nearest 10, with a floor of 20. If that summary is missing, N is the model's median. | Whether `kw_strat`'s coverage gain is extra length |
| `counts_neutral` | `refer_oracle` with its instruction removed: "{p} and {n} out of 8 are positive, negative." followed by the plain prompt. There is no "balanced" and no "reflecting this distribution". | Frequency information vs REFER's balance instruction |

**Hypotheses:** paired, one-sided, α = 0.05, Holm over 10 cells. Both use skewed levels only.
- **C1:** minority recall is higher under `kw_strat` than under `direct_len`.
  - Supported in a cell if the Holm-corrected p < 0.05.
  - We also report the achieved length difference (`n_words`, `direct_len` − `kw_strat`), so
    readers can see whether the length match held.
- **C2:** majority amplification is higher (closer to proportional) under `counts_neutral` than
  under `refer_oracle`. That is, the balance instruction itself pulls proportions toward even.
- **Descriptive only:** `counts_neutral` vs `direct` on MA, |Δp|, share stating counts, and count
  error.

## 3. Statistical unit

Entities repeat across levels:
- Amazon: 278 entities in 300 test collections (21 repeat).
- Drugs: 245 entities in 300 (49 repeat).

So, from this amendment on:
- Bootstrap intervals resample **entities** (product or drug), carrying all of that entity's
  collections.
- Paired tests are run on entity-mean paired differences.
- The registered collection-level results are re-run this way and reported side by side. Any
  verdict that changes is named.
- Effect sizes are mean paired differences with entity-clustered 95% intervals.
- Non-inferiority parts (H2c, H4c) also report TOST against the same margin. Each result is
  labelled *equivalent*, *non-inferior*, *inconclusive* or *worse*. A non-significant difference
  is never described as "no difference".

## 4. Descriptive analyses found after scoring

These are labelled as found after scoring.
- MA reported separately for positive-majority and negative-majority inputs, with the
  misclassification envelope drawn on the figure.
- Week-1 numbers that prompted this split (plain prompt, all models):
  - Positive majority: MA −0.125, well beyond the envelope.
  - Negative majority: MA −0.024, within it.
  - REFER: −0.169 and −0.107. `kw_strat`: −0.193 and −0.115.

## 5. Human evaluation, part B (whole summaries)

Part A (25 summaries × 4 statements, scorer diagnostic) is unchanged, including its α thresholds
and the rule that the first two annotators decide the primary faithfulness score.

**Sample**
- 80 collections from the skewed levels: 2 datasets × 4 levels × 5 models × 2 collections.
- Collections are drawn with seed 2026 from collections that meet all of these:
  - not in part A;
  - not the practice item;
  - both summaries are scored outputs (status ok or truncated).
- Each collection is shown with two summaries from the same model, `direct` and `kw_strat`, in
  random order, labelled A and B. Annotators are blind to condition and model.

**Procedure**
1. Both summaries are shown alone. For each: **Q2**, "From this summary alone, how do the
   reviewers feel overall?" The answer is one of five: almost all negative, mostly negative,
   evenly split, mostly positive, almost all positive.
2. The 8 reviews are revealed, grouped as positive (4–5 stars) and negative (1–2 stars). For each
   summary:
   - **Q1**, minority coverage: "How much of what the [minority-side] reviewers say does the
     summary convey?" None, some or most.
   - **Q3**, unsupported content: "Does the summary say anything no review supports, including a
     wrong number?" Yes or no, with an optional note.
   - **Q4**, readability: clear, minor problems, or hard to read.

**Annotators**
- Two pairs, each double-labelling 40 collections.
- If fewer than four annotators take part, the two primary annotators label all 80.

**Analysis**
- **Q1:** coded 0 / 0.5 / 1. Paired difference `kw_strat` − `direct`: mean, one-sided Wilcoxon,
  and entity-clustered bootstrap CI. Spearman correlation with automatic minority recall.
- **Q2:** each answer is coded as a positive share: almost all negative = 0.05, mostly
  negative = 0.25, evenly split = 0.5, mostly positive = 0.75, almost all positive = 0.95. That is
  converted to the share of the true majority side. "Human MA" is that perceived majority share
  minus the true majority share. Its sign is reported by
  direction and prompt, along with its agreement with classifier MA (correlation and mean bias).
- **Q3:** compared with automatic unsupported rate (AUC), under both the registered and the
  attribution-robust score.
- **Q4:** descriptive.
- **Agreement:** Krippendorff's α (ordinal for Q1, Q2 and Q4; nominal for Q3), with bootstrap CIs.
  A question with α below 0.667 is reported but not used to decide anything.

**Power**
- From the week-1 automatic scores:
  - Minority recall, `kw_strat` − `direct`: d_z = 0.33. This needs 73 pairs for 80% power, so 80
    pairs gives about 84%.
  - MA: d_z = 0.36, about 61 pairs.
- Faithfulness differences (unsupported rate, d_z = 0.09) would need about 950 pairs. Part B
  therefore does not test them. Faithfulness rests on part A's scorer validation and on the
  automatic non-inferiority tests.

## 6. How the paper is framed after the checks

- **If part B confirms the coverage gain** (Q1 difference > 0, CI excluding 0, Q1 α ≥ 0.667), the
  paper reports a practical coverage–proportion trade-off.
- **If it does not,** or human and automatic coverage disagree (Spearman < 0.3), the paper is
  about measurement error in opinion summarisation. The keyword results become a case study.
- Both outcomes are reported as they fall.

## 7. Text changes, not affecting any result

- Drug collections are described as preserving *patient-reported experiences*. They establish
  neither drug causality nor clinical safety. A positive review can still report a side effect.
- The registered faithfulness score and the attribution-robust score stay side by side. The
  latter is labelled as a documented amendment.
- The introduction states only what has survived these checks.

## Review points and what this amendment does about each

| Review point | Action |
|---|---|
| Two paper stories | §1: keyword paper only; FairSum is a separate paper and is not mentioned here |
| Cross-model analysis pending | Already done: 5 models, 27,000 summaries, no duplicates, failures logged in the compliance table |
| Human check thin | §5: part B, power-planned, double-labelled, whole summaries |
| Attribution sensitivity | Both scores kept side by side; part B Q3 checks both |
| Length-matched control | §2 `direct_len` (complements the regression in amendment 7) |
| Neutral-counts control | §2 `counts_neutral`, minimal pair with `refer_oracle` |
| Clustering by entity | §3 |
| Primary outcome, family, equivalence | §1, §3 |
| Opinion vs medical evidence | §7 |
| False balance only if validated | §1, §4, part B Q2 |
