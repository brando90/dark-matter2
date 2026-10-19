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
