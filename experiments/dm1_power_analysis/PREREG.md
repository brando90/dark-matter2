# DM-1: power analysis for a second ("dark matter") power-law term

Pre-registered 2026-10-19, before any run. Changes after the first run are listed at the bottom with the reason.

## Question
How long and how dense must a model-size ladder be, at realistic loss noise, before the standard tests can tell
L(N) = E + A N^-alpha + B N^-beta apart from L(N) = E + A N^-alpha, without inventing a second term when there is none?
And when the second term is real but missed, how wrong is the small-to-large extrapolation?

## Convention
alpha is the slow term (dominates at large N), beta > alpha is the fast "dark matter" term that only matters for
small models. (The journal used both beta < alpha and beta > alpha; this fixes beta > alpha.)

## Ground truth
- Base: E = 1.69, A = 406.4, alpha = 0.34 (Chinchilla's parameter term).
- Dark matter strength s = (B N_ref^-beta) / (A N_ref^-alpha) at N_ref = 1e6, s in {0, 0.03, 0.1, 0.3, 1.0}.
- beta in {0.68, 1.0} (2x and ~3x alpha).
- Ladder windows [N_min, N_max]: [1e5, 1e7] (tiny CPU ladder, DM-3), [1e7, 1e9] (Pythia-like),
  [1e6, 1e9] (Ludwig/Chinchilla-like), [1e5, 1e9] (ideal 4 decades).
- Points per decade: 2, 4, 8 (log-spaced, both ends included).
- Noise: additive Gaussian on loss, sigma in {0.005, 0.01, 0.02} nats (seed-to-seed spread of real ladders is in this
  range; assumption, to be checked against real ladders in DM-2).
- Replicates: 100 per cell with s > 0, 400 per null cell (s = 0).

## Fits and tests
- 1-term (E, A, alpha) and 2-term (E, A, alpha, B, beta = alpha + delta, delta > 0) by nonlinear least squares,
  multi-start, bounded.
- Wald: z = B_hat / SE(B_hat), SE from sigma_hat^2 (J^T J)^-1 in natural parameters at the 2-term fit.
- BIC with Gaussian likelihood: BIC = n ln(RSS/n) + k ln n; dBIC = BIC_1 - BIC_2.
- Condition number of J^T J (columns scaled to unit norm) for the 2-term fit: sloppiness.
- Ladders with <= 6 points are marked untestable for the 2-term fit (need >= 2 residual dof); the 1-term fit is used.
- **Primary detection rule (fixed now): z > 2 AND dBIC > 2.** Also reported: each test alone.
- Extrapolation: predict L at 100 x N_max with the 1-term fit and with the model the primary rule selects; error vs truth.

## Pre-registered success criteria
1. Positive control: in [1e5, 1e9], 8/decade, sigma = 0.005, s = 1, beta = 1, power >= 0.95 and median |beta_hat - beta|
   / beta <= 0.10. If this fails the fitter is broken and nothing else is reported.
2. False positives: primary-rule rate in every null cell <= 0.10, pooled <= 0.05. If not, the rule is too loose and
   the report says so.
3. Main readout: for each window, the smallest s detected with power >= 0.8 (at sigma = 0.01, 4/decade).
4. Prediction from Rylan's idea, written now: a [1e7, 1e9] ladder will have power < 0.5 for s <= 0.1 at sigma = 0.01,
   i.e. a Pythia-scale ladder cannot see a term that is 10% of the main term at 1M parameters; small models are needed.

## Changes after the first run
1. Wald SE inversion (Oct 19, found in DM-2 before the final analysis). The first run inverted the unscaled J^T J with
   pinv; when every n >> 1 and beta is large, the n^-beta column is ~1e-6 and pinv's cutoff dropped it, so SE(B)
   collapsed and z exploded. The formula is unchanged (sigma_hat^2 (J^T J)^-1); it is now inverted with unit-norm
   columns (dm1.wald). analyze.py recomputes z and the decisions that depend on it from each replicate's stored fit;
   117 of 36,000 testable primary decisions flipped. The raw CSVs keep the first-run z.
2. Where it ran. Cells 0-161 and part of 162 ran locally (results/raw_shard000_of001.csv); the box got busy, so cells
   162-323 ran on Sherlock's dev partition (job 48509059, results/raw_shard001_sherlock.csv). Replicates are seeded by
   (cell, rep); the 1,664 replicates in both files agree on every decision (floats to ~1e-12), and the local copy is
   kept.

# DM-1b addendum: a calibrated likelihood-ratio test in place of the Wald arm

Pre-registered 2026-10-19, after DM-1's results and before any DM-1b run.

## Why
DM-1's primary rule (z > 2 AND dBIC > 2) is limited by its Wald arm: the 5-parameter fit is sloppy, so the asymptotic
SE of B is large even when the fit recovers beta, and Wald alone has a worst null cell of 0.245. BIC alone finds terms
3-10x weaker but its worst null cell is 0.117. Neither threshold is calibrated to the ladder at hand. DM-1b calibrates
the likelihood ratio per ladder by simulation.

## Test (fixed now)
- Same 324 cells, same replicates and the same data: each replicate's y is regenerated from DM-1's seed
  (cell, rep, 20261019), and the replicates with <= 6 points stay untestable (the 1-term fit is used), as in DM-1.
- Statistic: T = n ln(RSS_1 / RSS_2), the Gaussian likelihood ratio of the nested fits (RSS_2 <= RSS_1 always).
- Null distribution, parametric bootstrap under the fitted 1-term null: y*_b = E1 + a1 n^-alpha1 + eps_b,
  eps_b ~ N(0, sigma_hat^2), sigma_hat^2 = RSS_1 / (n - 3), b = 1..B, B = 200, bootstrap seed (cell, rep, 20261019, 1).
  Each y*_b is refit with both laws and gives T*_b.
- p = (1 + #{T*_b >= T}) / (B + 1); **detect when p <= 0.05** (alpha = 0.05; the exact size under a known null is
  10/201 = 0.0498).
- Fitting: the observed data and every bootstrap sample are fit by the same vectorized variable-projection fitter
  (dm1b.py: grid over alpha, and over (alpha, delta) on DM-1's grids, then a shrinking local grid search; E, a, b >= 0;
  RSS_2 = min(RSS_1, best 2-term RSS)), so T and T* are computed identically. Its RSS_1 and RSS_2 on the observed data
  are compared with DM-1's stored fits and the agreement is reported.

## Checks (verify_dm1b.py)
Completeness (every planned (cell, rep) once), finite fields, nesting, determinism (recompute 5 replicates), and the
DM-1 criteria applied to the LRT:
1. Positive control: [1e5, 1e9], 8/decade, sigma = 0.005, s = 1, beta = 1: power >= 0.95 and median
   |beta_hat - beta| / beta <= 0.10 among detections.
2. False positives: LRT rate in every testable null cell <= 0.10, pooled <= 0.05. Written now: a correctly calibrated
   test sits at ~0.050 pooled by design (12,000 null replicates, Monte Carlo SE ~0.002), so this bar is tight; it is
   applied literally, and a pooled rate in (0.050, 0.054] is also reported as within Monte Carlo error of nominal.

## Readout
For each window and beta: smallest s with LRT power >= 0.8 at sigma = 0.01, 4/decade, next to DM-1's primary rule and
BIC alone on the same replicates; LRT power on the full grid; median 100x-extrapolation error of the LRT-selected model.

## Predictions (written now)
1. The LRT's null rate is near 0.05 in every testable cell (it is calibrated per ladder), unlike BIC's 0.117 worst cell.
2. The LRT's power is close to BIC alone's (both use RSS_1 / RSS_2) and well above the primary rule's, so the min
   detectable s drops by >= 3x vs the primary rule in at least two of the four windows.
3. [1e7, 1e9] still cannot see s <= 0.1 (power < 0.5): the limit there is the data, not the test.

## DM-1b: changes before the main run
1. Fitter (Oct 19, after smoke and timing runs on 1% of the replicates, 432 of 43,200; no test setting changed).
   The first version (grid, then a shrinking local grid) missed DM-1's 2-term optimum by > 1% RSS on 6.7% of
   replicates: the RSS surface has narrow curved valleys and optima on the bounds (delta at 4, the fast term fitting
   only the smallest models; alpha near 0 with E = 0). The fitter now runs a compass search from five starts (the best
   grid point in each of three delta bands, plus those two boundary points) and polishes each end point with
   projected Levenberg-Marquardt in all five parameters, keeping the best; on the same 360 testable replicates it is
   > 1% worse than DM-1 on 0.3% and > 0.1% better on 4.4%. The observed data and every bootstrap sample still go
   through the same fitter.
2. Exact shortcuts that leave every decision unchanged: a replicate with T = 0 has p = 1 (every T* >= 0), so it is
   not refit; and after the first 50 bootstrap samples a replicate with >= 10 exceedances can no longer reach
   k <= 9 of 200, so it stops there (its p is then reported from the samples used, column b_used). Bootstrap sample b
   is the same draw whichever stage uses it.
