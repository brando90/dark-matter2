# DM-2: is there a second ("dark matter") power-law term in real model-size ladders?

Pre-registered 2026-10-19, before any fit. The data were opened only to learn their structure (which runs, sizes,
token budgets and columns exist). Changes after the first fit are listed at the bottom with the reason.

## Question
Do public ladders support L = E + A N^-alpha + B N^-beta (beta > alpha, the fast term that matters only for small
models) over L = E + A N^-alpha? Does the answer survive the parameter-count convention (H5), the token-budget
confound, and out-of-sample checks (hold-out, reverse ladder)?

## Convention (same as DM-1)
alpha is the slow term, beta = alpha + delta with delta > 0 is the fast term. E, A, B (and G below) >= 0.
Fits work in n = N / 1e6 and d = D / 1e9 tokens. Least squares on loss in nats, as in DM-1, so DM-1's power table
applies directly.

## Two structural facts found while reading the data (before fitting)
1. **"Final loss per size" is confounded by tokens per parameter.** Porian's single-run curves end at ~1,250 tokens
   per parameter for the 5M model and ~5 for the 901M model; the ColPret LLaMA and Hoffmann Chinchilla sets are IsoFLOP
   designs with several token budgets per size. A last-checkpoint loss per size mixes an N effect with a D effect.
2. **Even a fixed-ratio slice is a two-term law in N.** With L(N, D) = E + A N^-alpha + G D^-gamma, the slice D = kN
   gives E + A N^-alpha + G k^-gamma N^-gamma: an N-only fit on it can find a "second term" that is the data term.

So DM-2 has two analyses.

### A. N-only, one curve per (dataset, hparams, schedule): the acceptance table
Per size, y = loss at D = 20 N (N as the source counts it, for the ratio only):
- several annealed runs per size (Porian `chinchilla` schedule = cosine to each budget; Hoffmann Chinchilla):
  linear interpolation of final loss in log(tokens/param) between the two runs that bracket 20; a size with no
  bracketing pair is dropped;
- one run per size (Porian `kaplan` and `const` schedules, LLaMA): the run's validation curve read at the step where
  tokens = 20 N (interpolated in log step); a size whose run ends before 20 N is dropped. For `kaplan` and LLaMA the
  learning rate is part-way through its decay at that step, by a different amount per size: labelled
  schedule-confounded.
- Confound check, labelled as such: y = the last checkpoint of the longest run per size (the literal "final loss").

Curves: Porian {rw, owt2} x {base/short/chinchilla, base/short/kaplan, base/long/kaplan, tuned/short/const,
tuned/long/const}; Hoffmann Chinchilla (Epoch's extraction, 245 points, via Rylan Schaeffer's robustness repo);
ColPret LLaMA (MAD, 21 sizes). Porian's `seed` runs give seed sigma only; its `sweep` runs are not used.

### B. Joint (N, D): the decisive test
On annealed final losses only (one point per run): Porian `chinchilla` schedule (rw, owt2; ~100 runs each),
Hoffmann Chinchilla (245), LLaMA run finals (last checkpoint of each run; assumes each run's schedule ends at its
budget, as in the MAD IsoFLOP design). Model: E + A n^-alpha + B n^-beta + G d^-gamma vs the same with B = 0.

## Parameter counts (H5)
- Porian: exact from the architecture (OpenLM, SwiGLU, hidden h = 256 * ceil(int(8w/3) / 256), tied embedding):
  N_nonemb = depth * (4 w^2 + 3 w h), N_total = N_nonemb + 50432 w. verify.py checks N_total against the checkpoint
  names (params=611M etc.) within 3% (names are rounded). The repo's width x depth is not used.
- Chinchilla: Rylan's table (d_model, n_layers, vocab 32,168): N_total = tied embedding + layers x (QKV + out + FFN),
  N_nonemb = layers x (QKV + out + FFN); Hoffmann's reported N kept as a third variant (Rylan: off ~7%, up to 15%).
- LLaMA: only the total in ColPret (no architecture), so H5 cannot be run there; said so in the result.
Primary count: N_total. Every fit is repeated with N_nonemb (and N_reported for Chinchilla).

## Fits and tests
- 1-term and 2-term fits: DM-1's variable projection (dm1.py fit1/fit2, copied verbatim into fitlib.py), grid over
  the exponents, linear solve for the coefficients, bounded refinement. The joint model uses the same method with
  gamma as one more nonlinear parameter and G as one more linear coefficient.
- Wald z = B / SE(B), SE from sigma^2 (J^T J)^-1 in natural parameters (DM-1's code).
- dBIC = BIC_1 - BIC_2 with Gaussian BIC (n ln(RSS/n) + k ln n).
- Condition number of J^T J with unit-norm columns at the 2-term fit.
- Bootstrap SE(B): residual bootstrap, 1,000 refits: y* = fitted 2-term curve + centred residuals resampled with
  replacement; SE_boot = sd of B*. Also the share of refits with B* > 0 and z_boot = B / SE_boot.
- Seed sigma: per (dataset, size) the sd of final loss across Porian's seed runs (7-8 seeds; widths 96, 128, 160 at
  ~20 tokens/param, 224, 320, 480 at ~200); reported per size and pooled (rms).
- **Primary detection rule (fixed now, = DM-1): z > 2 AND dBIC > 2.** A curve "shows a second term" only if it also
  has z_boot > 2. Each test is also reported alone.
- Hold-out: sort sizes, fit the smaller ceil(2/3), predict the rest; RMSE of 1-term vs 2-term prediction.
- Reverse ladder: fit the 1-term law on sizes >= 100M, predict the smaller sizes; report each residual
  (observed - predicted), their mean, and how many are positive. Dark matter predicts positive residuals growing as N
  shrinks. Skipped (marked) when fewer than 4 sizes are >= 100M.
- For B (joint): hold-out and reverse ladder split by N the same way; all D at a size go together.

## Reading a null with DM-1
For each N-only curve: span (decades), points per decade, residual sigma of the 2-term fit and seed sigma; the
matching DM-1 cells (window [1e7, 1e9] when the curve's span is about 2 decades, nearest points/decade, nearest
sigma) give the power for s = 0.03 ... 1 and beta = 0.68, 1. A null on a curve whose matched power is < 0.5 is "not
detectable here", not "absent". DM-1's [1e6, 1e9] and [1e5, 1e9] windows are incomplete (AT-492); cells used are
from data/dm1_cells_partial.csv.

## Predictions, written now
1. Analysis A at D = 20N: no curve passes the primary rule (DM-1: a ~2-decade ladder above 5M has power < 0.5 for
   s <= 0.3 at sigma = 0.01).
2. The literal final-loss curves (tokens/param falling from ~1,250 to ~5) give larger dBIC than the D = 20N curves
   of the same runs: the token-budget confound, not dark matter.
3. Analysis B: B is not significant (primary rule) in at least 3 of the 4 joint ladders.
4. Reverse ladder on Porian at D = 20N: the mean residual of the sub-100M sizes is within +-0.05 nats.
5. H5: switching N_total to N_nonemb changes alpha by more than 0.05 on Porian (embedding share is ~95% at 5M and ~8%
   at 901M) and changes the verdict on at least one curve.

## Success criteria for the code (verify.py)
1. Positive control: a synthetic curve with the Porian sizes, E = 1.69, A, alpha = 0.34, s = 1, beta = 1, sigma =
   0.005 is detected (primary rule) and beta is recovered within 20%; the same with s = 0 is not detected.
2. Porian N_total matches the checkpoint names within 3% for all 16 sizes.
3. Every curve in the tables has the planned sizes (or the documented drops), and every bootstrap has 1,000 refits.
4. RSS_2 <= RSS_1 for every fit (nested models).
5. results/ hashes listed in results/manifest.txt match the files.

## Changes after the first fit
1. **Wald SE inversion (bug fix, applies to DM-1 too).** fit2's SE inverts the unscaled J^T J with pinv; on real
   ladders every n >> 1, so for large beta the n^-beta column is ~1e-6 and pinv's cutoff drops it: SE(B) -> 0 and z
   reached 1e8 on null fits. z is recomputed with unit-norm columns (fitlib._scaled_se, same formula). tableA keeps
   fit2's own z as z_wald_dm1code. On DM-1's 16,064 testable replicates this flips 69 detections; conclusions hold.
2. **Positive control** is run on 50 synthetic replicates (detection rate >= 0.9, median beta within 20%) instead of
   one draw, so one noise draw cannot pass or fail it.
3. **Porian name check:** the checkpoint names are floor(N / 1M), not rounded, so verify.py checks exact equality of
   floor(N_total / 1M) with the name (stronger than "within 3%").
4. **Post hoc, labelled as such:** the Chinchilla-form joint law misfits Porian's base-hparam `chinchilla` runs
   (residual sd ~0.21 nats), so analysis B could not test the one curve family with an N-only signal (tuned, constant
   LR). Added a joint fit on the tuned constant-LR validation curves read at D/N = 2.5..320 (tableB_explore.csv).
   Points from one run are correlated, so its SEs are optimistic.

---

# DM-3 addendum: embedding accounting vs a real second term (tuned Porian curves)

Written 2026-10-19 before any DM-3 fit. Known from DM-2 (so this is not blind): on Porian tuned-short-const rw and
owt2 at D = 20N the second term is significant with N = N_total (c = 1) and absent with N = N_nonemb (c = 0); the
Chinchilla-form joint law misfits the tuned constant-LR curves (residual sd 0.08-0.09).

## Question
Is the fast term on these two curves a real second power law, or is it what you get from counting an embedding
parameter as a full body parameter? Model the count as N_c = N_nonemb + c * N_emb (N_emb = 50,432 w, tied) and let
the data choose c.

## Data (unchanged from DM-2)
- A3, N-only: the DM-2 D = 20N values of porian/{rw,owt2}/tuned-short-const (14 sizes each, 5M-455M by N_total).
  y is held fixed; only the x-axis count changes with c (D stays 20 N_total).
- B3, joint: results/joint_points_explore.csv (the same runs' constant-LR validation curves at D/N_total = 2.5-320;
  rw 93 points, owt2 89, 16 sizes). Points from one run are correlated: SEs there are optimistic.

## A3: profile c
- Grid c in {0, 0.005, ..., 2.0} (401 values). At each c: DM-1's fit1 and fit2 (fitlib, unchanged) on n = N_c / 1e6.
  RSS1(c), RSS2(c), B(c), Wald z(c) (stable inversion).
- c1 = argmin RSS1 (1-term law + free c, k = 4); c2 = argmin RSS2 (2-term law + free c, k = 6). Each refined with a
  bounded scalar search around the grid minimum.
- 95% CI for c: profile likelihood, {c : n ln(RSS(c) / RSS_min) <= 3.84}, for both models; plus a residual bootstrap
  CI of c1 (1,000 refits of the 1-term-with-c model, percentile 2.5-97.5).
- Second-term test with c free (the decisive A3 test): dBIC = BIC(1-term, c1; k = 4) - BIC(2-term, c2; k = 6);
  Wald z of B at c2 (natural parameters, c held at c2, so slightly optimistic); bootstrap z = B(c2) / sd(B*), 1,000
  residual-bootstrap refits from the 2-term-at-c2 curve, with c re-profiled in every refit (grid of 21 values over
  [0, 2] + bounded refinement). Primary rule as DM-2: z > 2 AND dBIC > 2, and "shows a second term" also needs
  boot z > 2.
- Also reported: the A3 table at c = 0, c = 1 (DM-2's two counts, as a check they reproduce DM-2) and at c1.
- Hold-out (sizes sorted by N_total, fit the smaller 10 of 14, predict the largest 4): 1-term vs 2-term RMSE with c
  fixed at the full-data c1 (the count is a property of the architecture; labelled "c from all data"), and strictly
  out of sample with c re-profiled on the 10 training sizes.
- Reverse ladder (the 5 sizes with N_total >= 100M; 108M-455M): 1-term law with c = c1 fitted on them, residuals
  (observed - predicted) of the 9 smaller sizes. A 2-term law cannot be fitted on 5 points, so for the 2-term model
  the per-size residuals of its full fit at c2 are reported next to them. Strict version: c also profiled on the 5
  large sizes (weakly identified there; reported, not used for the verdict).

## B3: Kaplan-style joint law
Kaplan et al. (2020) eq. 1.5 with an irreducible term and an optional second N term:
  L = E + (A n^-p + G / d)^q [+ B n^-beta],  n = N_c / 1e6, d = D / 1e9, p = alpha_N / alpha_D, q = alpha_D,
  beta > p q (faster than the main N term).
- K0: E = 0 (Kaplan as published), K1: E >= 0 free, K2: K1 + B n^-beta. All coefficients >= 0. Nonlinear least
  squares (scipy least_squares on log-parametrised positives, 40 starts, best RSS kept; K2 also starts from the K1
  optimum with B ~ 0 so RSS_K2 <= RSS_K1).
- c profiled on a grid {0, 0.05, ..., 1.5} for K1 and K2 (refined around the minimum); 95% profile CI as above.
- Compared on residual sd against the DM-2 Chinchilla-form joint fit on the same points (0.08 rw / 0.08 owt2 with
  N_total) and refitted at the K1 best c.
- Second-term test in B3: Wald z of B (c held at the K2 best c), dBIC (K1 vs K2 at their own best c; k = 6 vs 8 with
  c counted), bootstrap z (1,000 residual-bootstrap refits, c fixed at the K2 best c, 8 starts per refit: cheaper,
  and labelled conditional on c).
- Hold-out by size (fit the smaller 11 of 16 sizes, all their D; predict the 5 largest) and reverse ladder (fit on
  N_total >= 100M, 7 sizes; mean residual per smaller size), both for K1 and K2 at their full-data best c.

## Decision rule for the verdict (fixed now)
- **Embedding accounting**, if on both datasets: (i) A3 with c free fails the primary rule or boot z <= 2, (ii) the
  95% profile CI of c1 excludes 1, and (iii) the A3 reverse-ladder mean residual at c1 is within +-0.05 nats
  (DM-2 at c = 1: +0.10). B3 must not contradict it (K2's B not significant at its best c, or K2's best c CI
  excludes 1).
- **Dark matter (survives the count)**, if on both datasets A3 with c free passes the primary rule with boot z > 2
  and the B3 K2 test agrees (B significant, z > 2 and dBIC > 2).
- Anything else: **undetermined**, with the reason (e.g. c not identified, the datasets disagree, A3 and B3 disagree).

## Predictions, written now
1. c1 is well below 1 on both datasets (point estimate < 0.5) and its CI excludes 1.
2. With c free the second term fails the primary rule on both datasets (embedding accounting).
3. The A3 reverse-ladder mean at c1 is within +-0.05 nats.
4. The Kaplan form with E (K1) fits the constant-LR curves better than the Chinchilla form (lower residual sd).
5. K2's second term is not significant at its best c.

## verify.py additions (DM-3)
1. c recovery: 50 synthetic curves on the Porian sizes, y = 1-term law in N_c with c = 0.3 (E 1.8, alpha 0.3,
   A from the rw c = 0.3 fit if available, else 15), sigma 0.005: median c1 within 0.1 of 0.3 and the profile CI
   covers 0.3 in >= 80%.
2. Null control with c free: on those same curves the c-free second-term test fires in <= 10%.
3. Kaplan fitter: a noiseless synthetic K1 surface on the B3 design (c = 0.5) is recovered: p, q within 5%, RSS < 1e-8.
4. The A3 profile at c = 0 and c = 1 reproduces DM-2's tableA (alpha1 and rss1 within 1e-6 relative) for both
   datasets.
5. Every bootstrap has 1,000 refits; RSS_2 <= RSS_1 (A3 at every c on the grid, B3 K2 vs K1 at the same c);
   manifest hashes match.
