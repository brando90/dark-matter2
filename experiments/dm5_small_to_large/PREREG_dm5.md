# DM-5 PREREG: small-to-large predictability (H2), rolling-origin extrapolation

Written 2026-10-21 (America/Los_Angeles), committed on branch `dm5-small-to-large` (base `dm2-real-ladders`) BEFORE any
DM-5 fit. No DM-5 loss has been fitted or predicted at the time of this commit; only the ladders' structure
(sizes per family, which columns exist) was inspected to write this file.

## Question (H2)
A 1-term law L = E + a n^-alpha fitted on small models over-estimates alpha and mis-extrapolates when a faster-decaying
second term is present (DM-1/DM-1b). Does a rule that handles the second term (selected 2-term, or dropping the smallest
sizes) predict the loss of LARGER held-out models better than the plain 1-term fit, at >= 10x extrapolation?

## Ladders (one y per size; n = N / 1e6, y in nats)
Qualifying = >= 6 sizes with a loss column (not ppl). DM-2's definitions are reused where they exist (results/curves.csv,
slice D = 20N).
| id | source | y | N (primary) | sizes |
|---|---|---|---|---|
| chinchilla-43 | ColPret chinchilla / Epoch Hoffmann points | loss at D = 20N (DM-2) | N_total (DM-2) | 43 |
| llama-21 | ColPret LLaMA | median train loss within +-5% tokens of 20N (DM-2) | num_params | 21 |
| hyena-19 | ColPret Striped Hyena 1-11 (unified csv) | same as llama-21 | num_params | 19 |
| gpt3-8 | ColPret gpt3 | last val_loss per model (all models see ~300B tokens: fixed D, NOT D = 20N; labelled) | num_params | 8 |
| porian-rw, porian-owt2 | Porian tuned-short-const | val loss at D = 20 N_total (DM-2 curve value) | untied count N = N_nonemb + 2 * 50,432 w (Amendment 1) | 16 each |
Secondary (reported, not in the verdict): porian-rw and porian-owt2 with N = N_nonemb (DM-3 convention c = 0), and with the
tied count N_total of DM-2.
Not qualifying (reported as such): overtrain (4-5 sizes per dataset), OPT/bloom/lambda/t5 (<6 sizes or ppl only), Pythia
(ppl only; no ln(ppl) conversion is done), GPT-2 families (3 sizes). Overtrain therefore has no fold in the primary design.

## Folds
Sort a ladder by N. For k = 4 .. n-1: fit on the smallest k sizes; predict EVERY larger held-out size (n-k targets). Row
per (ladder, k, rule, target). Extrapolation ratio r = N_target / N_max_fit.

## Rules (three, as ordered)
1. **1-term**: L = E + a n^-alpha, DM-1/DM-1b fitter (`dm1b.fit_batch`, variable projection + LM, E, a >= 0).
2. **LRT-selected 2-term**: T = k ln(RSS1/RSS2) with 2-term L = E + a n^-alpha + b n^-beta, beta > alpha; parametric
   bootstrap under the fitted 1-term null, B = 200 (sigma^2 = RSS1/(k-3)); use the 2-term prediction iff (1+j)/201 <= 0.05,
   else the 1-term prediction. Defined only for k > 6 (DM-1's testable rule); for k <= 6 the rule is the 1-term fit
   (flagged `lrt_tested = 0`).
3. **Drop-smallest**: 1-term fitted after dropping the smallest d = 1 (Drop1) or d = 2 (Drop2) of the k fit sizes;
   defined only when k - d >= 4. Both are reported; the verdict covers each separately.

## Outputs and statistics
Per-fold CSV (all rows), summary: median |error| (nats) by ladder, rule and ratio bin [1,3), [3,10), [10,30), [30,inf),
plus signed median (bias). Paired comparison: for each rule, over (fold, target) pairs where both rule and 1-term are
defined and r >= 10, the ladder-level median of |err_1term| - |err_rule| ("gain", positive = rule better).
## Verdict rule for H2 (fixed now)
For each rule: SUPPORTED if it has positive gain on >= 5 of the 6 primary ladders AND the median of the 6 ladder-level
gains is >= 0.005 nats (about the seed noise scale of DM-2); WORSE if 1-term wins (negative gain) on >= 5 of 6; otherwise
NULL (mixed). Gains are also reported as a percentage of the 1-term median |error|. Three rules are tested; no
multiplicity correction beyond the 5/6 requirement, and the report says so. Ladders with no pairs at r >= 10 are listed
and excluded from the count (then the threshold becomes all-but-one of those available).

## Checks (verify_dm5.py)
(a) fold counts equal sum over ladders of (n-1) - 3 targets-per-fold structure; (b) no NaN in any defined prediction;
(c) positive control: a planted ladder y = 1.8 + 3 n^-0.30 + 12 n^-1.2 with sigma = 0.005 on 14 sizes (1e5..1e9) must make
the LRT rule detect the second term in a majority of its k > 6 folds and beat 1-term at r >= 10; (d) a planted pure 1-term
ladder must give LRT false-detection rate <= 0.15 over its folds.

## Compute
Estimated well under 2 CPU-h (about 100 folds x 200 bootstrap refits with the vectorized fitter); a 1% timing sample
runs on a synthetic ladder first. If the full grid exceeds ~2 CPU-h on this box it goes to Sherlock normal CPU.

## Limitations stated in advance
Ladders differ in token budget (gpt3-8 is fixed-D), optimizer and tuning; Porian and Hoffmann were fitted by other
people with their own hparams. The y of the training-loss ladders (LLaMA, Hyena) is a noisy training loss, not a held-out
loss. A failure to beat 1-term is not evidence against a second term (DM-1: nulls only rule out a large term).
