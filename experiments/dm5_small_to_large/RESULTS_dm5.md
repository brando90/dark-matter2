# DM-5 results: small-to-large predictability (H2)

Run 2026-10-21 on this box (156 folds, 5,404 prediction rows, ~75 s on 3 CPUs; no Sherlock needed). PREREG_dm5.md
(commit 8395457) was committed before any fit. Code: dm5_data.py -> dm5.py -> summarize_dm5.py, verify_dm5.py (ALL PASS:
fold counts, no NaN, positive control: planted 2-term ladder detected in 56/56 folds and 1.13 -> 0.031 nats at >= 10x;
null control: false detections 1/56 = 0.018). Outputs in results/: `folds.csv` (all folds), `summary_by_ratio.csv`,
`gains_at_10x.csv`, `verdict_h2.csv`, `posthoc_*` (labelled), `dm5_error_vs_ratio.png`.

## Verdict on H2: NOT established. Mixed, and it lives in Porian.
By the fixed PREREG rule (>= 5 of 6 ladders better, median gain >= 0.005 nats; only 4 ladders have any pair at >= 10x, so
the bar was 3 of 4):
| rule | ladders better / worse (of 4) | median gain at >= 10x | verdict |
|---|---|---|---|
| LRT-selected 2-term | 0 / 0 | 0.000 nats | NULL (identical to 1-term there, see below) |
| Drop1 | 2 / 2 | -0.037 nats | NULL |
| Drop2 | 3 / 1 | +0.048 nats (+18%) | SUPPORTED by the rule, but fragile: Porian rw and owt2 contribute ONE fold (one k) each, gpt3 two pairs, chinchilla four |

**Read it as:** dropping the smallest two sizes helps a lot on Porian (the ladder where DM-3 found a second term), is a
wash on Chinchilla, and hurts LLaMA badly; the bootstrap-LRT 2-term never beats 1-term at >= 10x because it is almost never
allowed to differ there. Do not claim H2 as a general result.

## Headline table: median |error| (nats) by extrapolation ratio, primary ladders (N_pred / N_max_fit)
| ladder (sizes used) | rule | 1-3x | 3-10x | 10-30x | >=30x |
|---|---|---|---|---|---|
| chinchilla (28, N_total) | 1-term | .0196 | .0701 | .3030 | - |
| | LRT 2-term | .0240 | .1143 | .3030 | - |
| | Drop1 / Drop2 | .0167 / .0176 | .0677 / .0723 | .1667 / .1624 | - |
| gpt3 (8, fixed D) | 1-term = LRT | .0212 | .0405 | .0305 | .1012 |
| | Drop1 / Drop2 | .0299 / .0075 | .0869 / - | .0090 / .0499 | .2461 / - |
| hyena (13) | 1-term = LRT | .1316 | .3298 | - | - |
| | Drop1 / Drop2 | .1683 / .0709 | .5301 / .2434 | - | - |
| llama (15) | 1-term = LRT | .0645 | .0743 | - | - |
| | Drop1 / Drop2 | .0911 / .0983 | .2439 / .4127 | - | - |
| porian-owt2 (14, untied N) | 1-term | .0444 | .1706 | .4137 | - |
| | LRT 2-term | .0270 | .1706 | .4137 | - |
| | Drop1 / Drop2 | .0338 / .0111 | .1365 / .0269 | .4621 / .2257 | - |
| porian-rw (14, untied N) | 1-term | .0318 | .1288 | .3138 | - |
| | LRT 2-term | .0239 | .1288 | .3138 | - |
| | Drop1 / Drop2 | .0191 / .0040 | .0971 / .0305 | .4160 / .1372 | - |
Sizes are after dropping sizes with no D = 20N value (DM-2 convention): Chinchilla 43 -> 28, LLaMA 21 -> 15, Porian 16 -> 14,
Striped Hyena 19 -> 13 (same-N variants averaged). LLaMA and Hyena span < 10x after the k >= 4 fit sizes, so they have
nothing at >= 10x. The plot `dm5_error_vs_ratio.png` shows the same numbers (1-term is under the LRT line where they are
equal).

## Why the LRT rule is silent at >= 10x
Reaching 10x needs a small fit set, and DM-1's rule only tests with k > 6 points (5 parameters). At k <= 6 the rule IS the
1-term fit. Where it can fire (k >= 7, mostly 3-10x):
- Porian (tied or untied count): detects in 5 of 7 testable folds; when it fires the 2-term beats 1-term in every pair
  (post hoc, r >= 3: +0.03 to +0.07 nats untied, +0.08 to +0.13 tied; only 3-5 pairs each).
- Chinchilla: fires in 8 of 21 folds (9 of 21 with N_reported) and then LOSES: median gain -0.043 nats, better in 9% of pairs.
- LLaMA, Hyena, gpt3: never fires. Porian with N_nonemb: never fires (the DM-3 pattern: the second term vanishes with
  non-embedding counts).

## Post hoc (labelled, not in the verdict): ratio >= 3
All three rules NULL (3 of 6 ladders better for Drop1 and Drop2). Drop2: Porian rw/owt2 +0.077/+0.092 nats (+65%/+57%),
Hyena +0.098 (+29%); Chinchilla -0.023 (-44%), gpt3 -0.021, LLaMA -0.385 (-1119%: LLaMA's small sizes are on the curve
and dropping them throws information away). Secondary Porian counts: with the tied count Drop2 gives +0.14/+0.10 nats at
>= 3x; with N_nonemb Drop2 gives about 0 to +0.014 (the small-end waviness behind Drop2's gain is mostly an
embedding-accounting effect, consistent with DM-3's H5).

## Caveats
- Non-qualifying (reported, not fitted): overtrain (4-5 sizes per dataset), OPT/bloom/lambda/t5, Pythia (ppl only), GPT-2
  families (3 sizes). The task's "overtrain" ladder therefore has no fold.
- Training-loss ladders (LLaMA, Hyena) are noisy; gpt3 is fixed-D, not D = 20N; Chinchilla's usable range is 6e8 to 6e9 after the
  D = 20N filter (31x), so >= 10x folds are few for every ladder.
- 12 non-primary Porian/Chinchilla conventions are in folds.csv (primary = 0) and are secondary only.
- Bootstrap refits with no feasible solution (5 bootstrap samples in 2 of 156 folds) count as exceedances (conservative).
- Three rules tested with no multiplicity correction beyond the 5-of-6 (here 3-of-4) requirement; the Drop2 "SUPPORTED" would
  not survive a stricter bar on 1-fold Porian evidence. A real test of H2 needs ladders with >= 10 sizes spanning >= 100x
  with a clean loss column, which the public data barely has (DM-4's own ladder is built for this).
