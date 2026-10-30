# Part B of the human evaluation: implementation and analysis details

Keyword-prompting study. Builds on amendment 9 §5–6 (`2026-10-02-kw-amendment-9.md`) and amendment
10 §3 (`2026-10-05-kw-amendment-10.md`), which fix the sample, the questions, the analyses and the
rule for reading the result.

Written on 2026-10-30:
- after Part A was labelled and analysed;
- **before the Part B sample was drawn and before any Part B label existed.**

This note fixes only what those amendments leave open. Where it and they differ, they win.

## 1. Sample

- Test collections at the four skewed levels. Per (dataset, level) cell, 2 collections per model,
  10 per cell, 80 in all. No collection appears twice.
- Eligible: not one of Part A's 25 collections, not the practice collection `drugs-test-pos88-000`,
  and both the `direct` and the `kw_strat` summary of that model have status `ok` or `truncated`
  under the current scorer (version 4).
- Seed 2026. Cells and models are visited in sorted order, the model order is shuffled within each
  cell, and each model draws 2 collections from those still free in the cell.
- Which summary is A is drawn per collection under the same seed.
- Groups (amendment 10): the 80 sets are shuffled under the seed; the first 10 are the shared group,
  the next 35 group 1 and the last 35 group 2.
- Set ids `B01`–`B80` are assigned after a second shuffle. Their order says nothing about model,
  dataset or level, and each annotator's 45 sets mix the shared ones with their own group's.

## 2. Page

- **Stage 1:** summary A alone, then Q2 for A; then summary B alone, then Q2 for B. The topic is
  shown; the reviews are not.
- **Stage 2:** the 8 reviews, grouped as positive and negative by rating, with both summaries. Q1,
  Q3 (with an optional note) and Q4 for each summary.
- Once stage 2 is open, Q2 can no longer be changed. The server refuses a save that changes it.
- Q1 names the minority side: "How much of what the negative reviewers say does the summary
  convey?" (or positive). Answers: none, some, most.
- A practice set, the practice collection with the 3B model's two summaries, comes first. It is not
  analysed.

## 3. Units and aggregation

- A **judgment** is one annotator's answers for one summary. A **summary score** is the mean of its
  judgments' codes over the annotators who labelled it.
- Only annotators who completed every set assigned to them enter the analysis. Others are reported
  by name with their progress.
- Confidence intervals resample entities (the product or drug), 5,000 draws, as elsewhere in the
  paper. Agreement intervals resample sets.

## 4. Analyses

- **Q1 (coverage).** Codes none 0, some 0.5, most 1. Per set, the difference between the
  `kw_strat` and `direct` summary scores: mean, one-sided Wilcoxon (greater), entity-clustered CI.
  Spearman ρ between the 160 summary scores and the automatic minority recall as registered
  (`recall_minority`); the attribution-robust recall is reported beside it.
- **Q2 (perceived split), primary.** Each answer becomes a majority-side strength: "almost all" on
  the majority side 2, "mostly" 1, "evenly split" 0, "mostly" on the minority side −1, "almost all"
  −2. The true strength is 2 for 7:1 and 1:7, and 1 for 6:2 and 2:6. A judgment is more balanced
  than the truth if its strength is lower, about right if equal, more one-sided if higher. Shares
  are reported by prompt and majority side. **Sign test:** per set, the `kw_strat` minus `direct`
  difference of mean strengths; ties dropped; two-sided binomial test.
- **Q2, secondary and sensitivity.** The amendment-9 coding and the alternative coding turn each
  answer into a perceived positive share, then a perceived majority share. Human MA is that minus
  the true majority share (0.75 or 0.875). Reported by prompt and majority side, with Pearson and
  Spearman correlations against the classifier MA and the mean difference (human − classifier).
- **Q3 (unsupported content).** AUC of the automatic unsupported rate for telling judgments of
  "yes" from "no", over all judgments, under the registered and the attribution-robust score.
- **Q4.** Shares by prompt.
- **Agreement.** Krippendorff's α over summaries as units: ordinal for Q1, Q2 and Q4 (Krippendorff's
  ordinal metric over the pooled answer frequencies), nominal for Q3. Reported over all annotators,
  within each pair, and on the 10 shared sets across all four annotators.

## 5. Reading the result

Amendment 10 lists what the paper needs. Each item is decided as follows, and a question with α
below 0.667 cannot satisfy an item (amendment 9).

| Item | Holds when |
|---|---|
| (i) Part A | the attribution-robust score won the registered 0.05 rule (already decided) |
| (ii) Q1 | mean difference > 0, its CI excludes 0, and Q1 α ≥ 0.667 |
| (iii) Q2 | the sign test gives p < 0.05 with `kw_strat` more balanced, and Q2 α ≥ 0.667 |
| (iv) coverage agreement | Spearman ρ ≥ 0.3 with the registered minority recall |
| (v) Q3 | the AUC's CI excludes 0.5 under the primary faithfulness score, which Part A made the attribution-robust one; Q3 α ≥ 0.667 |

**Framing** (amendment 9 §6): the paper reports the coverage–proportion trade-off if (ii) holds
and ρ ≥ 0.3. Otherwise it is reframed as a study of measurement error, with the keyword results as
a case study. Either way every result is reported.
