# Amendment 10: second review: one more control, absolute-distortion check, human-evaluation fixes

Keyword-prompting study. This amendment builds on amendment 9 (`2026-10-02-kw-amendment-9.md`).

Fixed on 2026-10-05, in response to a second external review:
- after all registered and amendment-9 results were known;
- **before any `counts_reflect` summary was generated**;
- **before any human label existed** (both labelling routes showed 0 of 100).

## 1. One more control prompt: `counts_reflect`

`counts_neutral` (amendment 9) drops the whole clause "generate a balanced summary reflecting this
distribution". So C2 separates the counts from *the whole balancing instruction*; it does not
isolate the word "balanced". The new prompt removes that one word:

| Prompt | Wording before the plain prompt |
|---|---|
| `refer_oracle` (registered) | "{p} and {n} out of 8 are positive, negative, generate a balanced summary reflecting this distribution." |
| `counts_reflect` (new) | "{p} and {n} out of 8 are positive, negative, generate a summary reflecting this distribution." |
| `counts_neutral` (amendment 9) | "{p} and {n} out of 8 are positive, negative." |

Run details:
- all 600 test collections × 5 models, same decoding;
- stored in the controls run (`results/kw/controls/`).

**Test C3:** the majority gap is higher (closer to proportional) under `counts_reflect` than under
`refer_oracle` on skewed levels.
- Paired, one-sided Wilcoxon on entity means.
- Entity-resampled 95% interval.
- Holm across the 10 model × dataset cells of C3, the same family as C1 and C2.

**Descriptive:** `counts_reflect` vs `counts_neutral`, the effect of "reflecting this distribution"
without "balanced".

**How the paper reads it:**

| C3 | C2 | Reading |
|---|---|---|
| holds | holds | The word "balanced" itself pulls toward an even split. |
| fails | holds | The pull comes from the instruction as a whole, not the single word, and the paper says "the instruction to write a balanced summary reflecting this distribution". |

## 2. H2a is a mechanistic prediction, not a fairness guardrail

H2a ("majority amplification falls") was registered on the assumption that keywords amplify the
majority. Because summaries already under-represent the majority, passing H2a means *larger*
distortion. From now on:
- H2a is reported as the registered mechanistic prediction.
- The proportion check on the primary question is the **absolute** distortion: |majority gap| on
  skewed levels and |Δp| on all levels, `kw_strat` vs `kw_global` and `kw_strat` vs `direct`.
  - Mean paired difference, entity-resampled 95% interval, per model × dataset.
  - **Descriptive**, labelled as chosen after the results were known.

## 3. Human evaluation, fixed before any label

**Part A** (unchanged protocol):
- Every agreement figure gets a bootstrap 95% interval over statements. So does the difference in
  supported-versus-not agreement between the two faithfulness scores.
- The registered 0.05 rule stands, and the paper reports whether the interval of the difference
  excludes zero.

**Part B: sample and assignment.**
- 80 collections as in amendment 9, now split into 35 + 35 + 10.
- Each annotator pair labels its 35 plus the 10 shared collections, which all four annotators label.
- The 10 shared collections give cross-pair agreement and a check for pair-specific behaviour.
- Each annotator labels 45 collections.

**Part B: Q2 analysis.**
- **Primary (ordinal):** each perceived split is compared with the category the true split belongs to.
  - 7:1 and 1:7 → "almost all"; 6:2 and 2:6 → "mostly".
  - Each judgement is one of: more balanced than the truth, about right, or more one-sided.
  - Report the shares by prompt and direction, plus a paired sign test of `kw_strat` vs `direct`.
- **Secondary:** the amendment-9 numeric coding (0.05 / 0.25 / 0.5 / 0.75 / 0.95).
- **Sensitivity:** an alternative coding (0.125 / 0.25 / 0.5 / 0.75 / 0.875, matching the true splits).

**Part B: scope.**
- Part B compares `direct` with `kw_strat`. It validates the raw coverage gain, not the claim that the
  gain goes beyond length, which rests on the automatic analyses. The paper states this.
- Part B supports pooled conclusions only, not conclusions for individual models, datasets or levels.

**What the paper needs from the human evaluation:**
1. Part A: the attribution-robust score agrees with humans better than the registered score.
2. Part B Q1: annotators see more minority content in `kw_strat` than in `direct` summaries.
3. Part B Q2: annotators see the shift toward balance in the same direction.
4. Human and automatic coverage correlate at least moderately (Spearman ρ ≥ 0.3).
5. Part B Q3 is not disconnected from the automatic faithfulness scores.

If these hold, the paper reports the empirical result and the measurement contribution. If they do
not, it is reframed as a measurement study, as amendment 9 specifies.

## 4. Presentation (no effect on results)

- **Analysis-status table:** every analysis is listed as one of:
  - (a) registered before scoring;
  - (b) added later but fixed before the data it uses existed;
  - (c) descriptive, chosen after the results were known.
- **Size and precision:** the 1B and 3B models run in 16-bit and the 7–8B models in 4-bit. Every
  statement about model size says the two are confounded.
- **Coverage-measure limitation:** a review counts as covered if it entails any one statement, so a
  generic statement can cover several reviews. Part B Q1 is the direct check.
