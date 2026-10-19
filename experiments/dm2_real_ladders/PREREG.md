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
(none yet)
