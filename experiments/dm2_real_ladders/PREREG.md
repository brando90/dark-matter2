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

## DM-3 changes after the first fit
A smoke run (3 bootstrap refits, not kept) came before the full run. It showed the 1-term RSS profile in c has two
separate low regions (near c = 0.05, where E hits its bound 0, and toward the grid's top, c = 2). So:
1. The 95% profile interval is reported as a set of intervals (e.g. "[0.03, 0.08] U [1.6, 2.0+]"), and the decision
   rule's "CI excludes 1" means 1 is in none of them.
2. Post hoc, labelled: the 1-term RSS at c = 3, 5, 10, 30, 100, 1e4 and with N = N_emb alone (width only), to see
   whether the fall past c = 2 continues.
3. verify.py check 1 uses the rw 1-term fit at c = 0.3 for all three of E, A, alpha (not E 1.8, alpha 0.3) so the
   synthetic curves look like the data; checks 1-2 profile c on a 0.02 (1-term) and 0.04 (2-term) grid for speed.
4. Post hoc, labelled (dm3_posthoc.py, results/dm3_posthoc_K1_slice.csv): the fitted K1 surface evaluated along
   D = 20 N_total, put through the DM-2 reverse ladder, to ask whether the Kaplan data term alone reproduces the
   small-model excess.
5. The full run used 2 workers on a machine at load ~13 (4 cores): ~95 min instead of the planned ~45.

# DM-4 addendum: a ladder we train, varying embedding size at fixed body size (design; nothing run)

Written 2026-10-19, after DM-3 and before any DM-4 run, tokenizer or data download. Files:
experiments/dm4_embedding_ladder/ (design.py -> runs.csv, budget.csv, budget.md; power.py -> power.csv, power.md).
No training is launched until Brando confirms the GPU budget for that phase (see Budget and gates).

## Why this design
DM-3 could not pin the embedding weight c in N_c = N_nonemb + c N_emb: along one Porian ladder N_emb = 50,432 w
moves in lockstep with the body, so c, E and the second term trade off (95% set split in two, bootstrap z ~0).
DM-4 breaks that collinearity by changing N_emb at a fixed body (same width, depth, hyperparameters, data):
1. **tied vs untied** at a fixed tokenizer: the untied model adds V w input-embedding parameters at the same compute
   (the input lookup is ~free; the output matmul is the same) and the same tokens. This is the clean lever for c.
2. **vocab 8,192 / 32,768 / 50,432**: changes N_emb by 6x at a fixed body, but also changes the tokenizer, so each
   vocab gets its own intercept E_g and losses are put on one scale (Metric).

## Ladder (Porian's tuned shapes and hyperparameters, read from data/data_resolving_discrepancies)
| w | depth | N_nonemb | lr | bs (seqs of 2048) | beta2 | warmup tokens | seeds |
|---|---|---|---|---|---|---|---|
| 128 | 4 | 1.05M | 0.0115 | 28 | 0.99 | 7.50M | 3 |
| 160 | 5 | 1.74M | 0.0105 | 32 | 0.99 | 9.81M | 3 |
| 224 | 6 | 4.30M | 0.0090 | 44 | 0.99 | 15.6M | 3 |
| 288 | 8 | 7.96M | 0.0080 | 56 | 0.99 | 22.5M | 3 |
| 320 | 9 | 12.5M | 0.0074 | 64 | 0.99 | 28.7M | 3 |
| 384 | 10 | 17.7M | 0.0068 | 80 | 0.99 | 37.1M | 1 |
| 480 | 12 | 33.2M | 0.0059 | 104 | 0.99 | 57.4M | 1 |
| 576 | 14 | 55.7M | 0.0051 | 128 | 0.99 | 84.8M | 1 |
| 640 | 15 | 76.2M | 0.0047 | 160 | 0.99 | 108M | 1 |
| 704 | 18 | 113.5M | 0.0043 | 192 | 0.99 | 149M | 1 |

Hyperparameters are the same for every arm at a body (Porian tuned them for the 50k tied model); the LR check below
tests whether that biases the arms. Recipe as Porian (OpenLM): pre-norm, SwiGLU (hidden 256 ceil(int(8w/3)/256)),
rotary, qk-norm, z-loss 1e-4, independent weight decay 1e-4, AdamW beta1 0.9, grad clip 1.0, bf16 autocast,
context 2048, linear warmup then constant LR. Trainer: OpenLM pinned to one commit if it installs wheel-only on
Sherlock (torch 2.6.0+cu124), else a minimal trainer with this recipe; either way the anchor gate (G1) validates it.

## Arms
- Phase 1 (w128-w320), 3 seeds each: {8,192, 32,768, 50,432} x {tied, untied}. 50,432 is the GPT-NeoX tokenizer
  Porian used, so tied-50k is Porian's exact configuration (the anchor). 8,192 and 32,768 are byte-level BPE
  (HF tokenizers, NeoX's pre-tokenizer) trained on 1 GB of RefinedWeb from shards outside train and validation.
- Tokenizer-family control: own byte-level BPE at 50,432, tied, seed 0, at w128 and w224 (vs NeoX 50,432). It
  measures the offset between our BPE and NeoX's at equal vocab; it is reported, not fitted.
- Phase 2 (w384-w704), 1 seed: untied-50k (largest embedding) and tied-8k (smallest; 5% of N_total at w704).
  The tied-50k ladder above w320 is Porian's own rw tuned-short-const runs (w384-w1120, the sizes DM-3 used),
  pooled only if G1 passes. Own tied-50k at w480 checks the pipeline against Porian mid-ladder (G2).
- N_emb counts unique embedding parameters: tied V w, untied 2 V w. N_nonemb as DM-2 (no norms or biases).
- LR check (seed 0, constant LR, to D/N = 20): tied-8k and untied-50k at w128 and w224 with lr x0.5 and x2.

## Data, budget per body and seeds
- RefinedWeb (tiiuae/falcon-refinedweb), a fixed list of parquet shards (names and sha256 recorded in phase 0);
  validation = one held-out shard cut to the first 25M UTF-8 bytes of whole documents; tokenizer training = other
  shards.
- Every arm at a body sees the same text: D_bytes = 20 x N_total(tied-50k) NeoX tokens x the measured NeoX bytes per
  token. Token counts then differ by vocab (ASSUME 3.4 / 4.15 / 4.35 bytes per token for 8k / 32k / 50k).
- seed s sets the init and the document order; the same s gives the same byte order in every arm, so arm contrasts
  at a body are paired by seed.

## Metric
y = validation loss in **nats per NeoX reference token** = (sum of token NLL over the validation documents, nats) /
(number of NeoX tokens in those documents). For NeoX arms this is the usual loss; for the others it is their
nats per byte times a constant, so all arms share DM-3's scale and its +-0.05-nat thresholds keep their meaning.
Each arm packs the same documents into 2,048-token windows of its own tokenizer (so the context in bytes differs by
vocab; absorbed by E_g, listed under limitations).

## Schedules and annealed runs
- Main runs: constant LR after warmup to D/N = 20 (reference tokens, as bytes), evaluated at D/N = 2.5, 5, 10, 20
  (as DM-3's constant-LR points).
- Annealed (WSD cooldown) branches from the constant-LR checkpoint at 0.8 D_k, LR linear to 0 over 0.2 D_k:
  D/N = k in {5, 10, 20} for all six phase-1 arms (seed 0); k = 20 for both phase-2 arms. They replicate the
  verdict on annealed losses and give an annealed joint (N, D) slice.

## Analysis (fixed now)
Points: seed-mean y per (body, arm) at D/N = 20 from the constant-LR runs (primary, DM-3's convention); the
cool@20 losses repeat everything as a replication.
- M1: y = E_g + A n_c^-alpha;  M2: M1 + B n_c^-beta, beta > alpha;  n_c = (N_nonemb + c N_emb) / 1e6,
  g in {8k, 32k, 50k} (k = 6 and 8 with c counted). Unweighted least squares by variable projection (grids over c in
  [0, 3] step 0.01, alpha, beta - alpha; non-negative linear solve for E_g, A, B; bounded refinement), as DM-1..3.
  Weighted by 1/sd^2 (seed sd per body) as a sensitivity analysis only.
- T0 (model-free): at each phase-1 body and vocab, untied minus tied, paired by seed (3 pairs), mean and t-interval.
- P1, c identified: the 95% profile set of c under M2 ({c : n ln(RSS2(c)/RSS2_min) <= 3.84}) is one interval of
  width <= 0.5.
- P2, second term with c pinned: at M2's best c, Wald z(B) > 2 AND dBIC(M1 vs M2) > 2 AND bootstrap z > 2
  (1,000 refits; seeds resampled per point where there are 3, residual bootstrap elsewhere; c re-profiled in each
  refit on a 0.05 grid + refinement).
- P3, reverse ladder: M1 at its best c fitted on bodies with N_nonemb >= 17.7M (w384 up: phase 2 + Porian), mean
  residual (observed - predicted) over all phase-1 points, and the share that are positive.

## Decision rule for the verdict (fixed now)
- **Embedding accounting**: P1 holds, P2 fails, and the P3 mean residual is within +-0.05 nats.
- **Dark matter (survives a pinned count)**: P1 holds, P2 passes, and the P3 mean residual is > +0.05 nats with
  >= 80% of phase-1 points positive.
- Anything else: **undetermined**, with the reason (c not identified, P2 and P3 disagree, const and annealed
  disagree, a gate failed).
- After phase 1 only, the result is an interim c statement (c's 95% set under M1 and M2, T0), never a dark-matter
  verdict: P3 needs phase 2.

## Gates
- G1 (after phase 1a, w128 and w160): our NeoX tied-50k seed mean at D/N = 20 is within +-0.05 nats of Porian rw
  tuned-short-const at D = 20N (4.796 at w128, 4.598 at w160; DM-2 values). If not, stop and fix the pipeline;
  Porian's runs are not pooled until it passes on all five phase-1 bodies (4.335, 4.169, 4.045 at w224-w320).
- G2 (phase 2): own tied-50k at w480 within +-0.05 nats of Porian (3.771).
- LR check: an arm whose best LR multiplier beats x1 by more than 2 seed sd at w128 or w224 is flagged; its
  phase-1b runs then use that multiplier (budget re-issued before launch).
- Budget: each phase is launched only after Brando confirms its GPU-hours; phase 0 re-issues budget.md from measured
  throughput first.

## Predictions, written now
1. G1 passes.
2. T0: untied beats tied at 50k on every phase-1 body, but by less than c = 1 predicts.
3. c's 95% set (M2) lies inside [0.05, 0.8] and excludes 1.
4. P2 passes on the full design (the fast term survives a pinned c); P3 mean residual > +0.05 nats.
   So the predicted verdict is dark matter, held with low confidence (~55%).
5. cool@20 losses are 0.05-0.15 nats below the constant-LR values and give the same verdict.

## Budget (design.py, L40S, FLOP-based: 6 N_nonemb + 6 V w + 6 depth 2048 w per token; ASSUME MFU 6-15% at
w <= 224, 12-25% at w288-384, 20-35% above; replaced by measured throughput in phase 0)
| phase | widths | runs | GPU-h pessimistic / central / optimistic |
|---|---|---|---|
| 0 | install, tokenizers, data (CPU); 200-step throughput probe per body x vocab (dev) | - | <= 0.5 |
| 1a | 128, 160 | 77 | 4.5 / 2.7 / 1.8 |
| 1b | 224, 288, 320 | 113 | 30.6 / 19.8 / 14.0 |
| 2 | 384-704 | 21 | 63.3 / 45.1 / 35.9 |
| all | | 211 | 98.5 / 67.6 / 51.7 |
The largest single run is tied-8k at w704 (~9 GPU-h central: 8k needs the most tokens for the same bytes).
On H100 the times are ~2.5x shorter. With the current 8 GPU-h weekly cap, phases 0 and 1a fit in one week and
phase 1b takes about three; phase 2 needs a larger allocation (e.g. Marlowe) or about six weeks of the cap.

## Sherlock job-array plan
- Everything under $SCRATCH/dm4 with its own venv (module python/3.12.1; wheel-only: torch 2.6.0+cu124, numpy
  2.2.6, tokenizers, pyarrow; built in an `srun -p dev` step, never a shared venv).
- Prep (CPU; an `srun -p dev` step inside submit_dm4.sh, as Exp 18 r2): fetch the shard list and check sha256, train
  the two BPEs, pretokenize train and validation per tokenizer into uint16 memmaps (all three vocabs < 65,536), write
  bytes-per-token. ~15 GB text, ~20 GB tokens.
- Phase 0 probe: one `srun -p dev -G 1` step, 200 steps per (body, vocab), writes tokens/s; design.py re-run with it.
- Training: one array per phase, `sbatch -p gpu -G 1 --array=<ids>%4 train_dm4.sbatch`, where a task's index is
  a const run's array_index in runs.csv; the task runs the const run and then, in the same allocation, its cooldown
  branches (from checkpoints saved at 0.8 D_k), so no cross-job dependency. Small bodies pack the three seeds of
  one arm into one task. --time = 1.5x the measured estimate (48 h max), checkpoints every 30 min with --requeue.
  %4 caps concurrent GPUs at four.
- Post: an afterany CPU job gathers eval JSONs (y at each D/N, bytes and tokens seen, GPU-seconds) into
  $SCRATCH/dm4/dm4_phase<k>_results.tgz, fetched with ?download=1; analysis runs locally (dm4.py, written before
  phase-1a results are opened, plus verify.py DM-4 checks).
- submit_dm4.sh takes --phase and --approved-gpuh and refuses to submit when the phase's estimate is above the
  approved number. Slurm mail END,FAIL to both of Brando's addresses (his Sherlock default).

## verify.py additions (DM-4, before results are opened; 1-2 are in dm4_embedding_ladder/verify_design.py)
1. runs.csv regenerates byte-identical from design.py; N_nonemb and tied N_total match DM-2's porian_counts.
2. The power check (power.py, below) meets: under the 1-term laws, median c within 0.05 of truth and profile-set
   coverage >= 80%; second-term false-fire rate <= 10%; under DM-3's 2-term law it fires in >= 80% on the full
   design.
3. At analysis: every run reached its byte target; no NaN; G1/G2 recomputed; RSS2 <= RSS1 at every c; 1,000
   bootstrap refits; results manifest hashes match.

## Limitations, known now
Only RefinedWeb (no owt2); hyperparameters tuned for untied-50k only (Amendment 1; the LR check tests tied-8k and tied-50k); vocab also
changes context in bytes and tokenizer quality, which only E_g absorbs; seed counts at phase 2 are 1, so P3 rests on
the small bodies' seed means.

## Power check (power.py, run before this addendum was committed; CPU only, no training)
200 simulated datasets per cell on the DM-4 design (constant-LR seed means at D/N = 20; Porian seed sd per width,
the larger of rw/owt2, divided by sqrt(seeds)); planted laws are DM-3's rw fits; fitted with M1/M2 above on a
0.05 c grid (the analysis uses 0.01 + refinement). verify_design.py: ALL PASS.
| design (points) | planted law | median c (M1) | M1 95% set covers c | second term fires | median dBIC |
|---|---|---|---|---|---|
| phase 1 (30) | 1-term, c = 0.1 | 0.10 | 1.00 | 0.00 | -6.8 |
| phase 1 (30) | 1-term, c = 0.5 | 0.50 | 0.96 | 0.045 | -6.6 |
| phase 1 (30) | 1-term, c = 1.0 | 1.00 | 0.92 | 0.01 | -6.4 |
| phase 1 (30) | DM-3 2-term, c = 1.07 | 1.05 (M2) | - | 1.00 | 59 |
| phase 1+2+Porian (48) | 1-term, c = 0.1 | 0.10 | 1.00 | 0.045 | -7.7 |
| phase 1+2+Porian (48) | 1-term, c = 0.5 | 0.50 | 0.96 | 0.065 | -6.6 |
| phase 1+2+Porian (48) | 1-term, c = 1.0 | 1.00 | 0.92 | 0.00 | -5.8 |
| phase 1+2+Porian (48) | DM-3 2-term, c = 1.07 | 1.05 (M2) | - | 1.00 | 178 |
So the design pins c to about +-0.05 (DM-3's split [0.05, 0.11] U [1.55, 2.0+] collapses to one point-like set),
keeps the second-term test's false-fire rate <= 6.5% and detects DM-3's 2-term law every time. Under the 2-term law
M2's set covers the true c = 1.066 in only 27-48% because it lies between grid points and the set is one grid point
wide: a grid artefact the 0.01 analysis grid removes. These numbers assume the law form is right, the same law in
every tokenizer group up to E_g, and Gaussian seed noise; a tokenizer effect that changes with size is not modelled.

## DM-4 Amendment 1 (2026-10-20, phase 0, before any DM-4 training run)
Phase 0 read Porian et al.'s released checkpoints (huggingface.co/formll/resolving-scaling-law-discrepancies) and
their evaluating_checkpoint.py. Three facts differ from what DM-2, DM-3 and the DM-4 addendum assumed:
1. **Porian's models are untied.** evaluating_checkpoint.py builds every model with `weight_tying: False`, and the
   w128 checkpoint (rw, tuned, short warmup, constant LR, step 2,421) has `output.weight` != `tok_embeddings.weight`.
2. They use 4 attention heads at every width (head dim w/4), LayerNorm with gain and bias (also as qk-norm over the
   full width), rotary on the full head dim; SwiGLU hidden 256 ceil(int(8w/3)/256) as assumed.
3. Their N (body + head, D = 20N) is N_nonemb + 50,432 w: the head is counted, the input embedding is not. DM-4's
   data per body (20 x N_total of "tied-50k" = 20 (N_nonemb + 50,432 w)) is therefore unchanged; it is renamed N_ref.

Changes to the DM-4 design (design.py, runs.csv, power.py), made now, before any run:
- The anchor arm is **untied-50k** (NeoX), Porian's exact configuration. G1: our untied-50k seed mean at D/N = 20 is
  within +-0.05 nats of Porian at w128/w160 (4.796/4.598); thresholds and rule unchanged.
- Phase 2 runs untied-50k (own; its w480 run is G2 against Porian's 3.771) and tied-8k. The extra own tied-50k run
  at w480 is dropped (the untied-50k w480 run is the same check): 210 runs instead of 211. Porian's own untied-50k
  runs at w384-w1120 add to the untied-50k arm, pooled only if G1 passes; they enter the fit with N_emb = 2 V w.
- LR check: tied-8k and tied-50k at w128 and w224 (x0.5, x2); untied-50k is the tuned configuration itself.
- Hyperparameters were tuned by Porian for untied-50k, not tied-50k; the limitations line reads accordingly.
- power.py's Porian points are untied (N_emb = 2 V w); re-run results below.
- Byte-level BPE: one HF tokenizers run at 50,432 on the 1 GB slice (NeoX's normalizer and pre-tokenizer); the 8,192
  and 32,768 tokenizers are its first V - 257 merges (BPE is greedy, so a smaller run is this prefix).
- Training data: shards are consumed in index order, each shuffled by (seed, shard), so a run's bytes never depend
  on how many shards are prepared; phase 0/1 prepares 8 training shards (~3.8 GB; w320 needs ~2.5 GB), phase 2 adds
  shards before its launch. The bytes-per-token that sets D_bytes is NeoX's on the first training shard (fixed
  whatever the pool size).
- Trainer: a minimal re-implementation of OpenLM's model (model.py; parameter names identical, so Porian's
  checkpoints load strictly) and recipe (train.py). eval_porian.py scores Porian's own w128/w160 checkpoints on our
  validation slice: that checks the model code and gives the validation-set offset, reported next to G1 (not a gate).

Consequence for DM-3 (not re-run here): DM-3 counted Porian's embedding as N_emb = 50,432 w (tied). With the true
untied count 2 x 50,432 w, every c DM-3 reports is in units of 50,432 w; in unique embedding parameters it is c/2
(DM-3's split set [0.05, 0.11] U [1.55, 2.0+] becomes [0.025, 0.055] U [0.78, 1.0+]). DM-3's verdict (undetermined)
does not change, since along one ladder the count only rescales c.

Power check re-run (power.py --jobs 4: the same random streams; phase-1 rows reproduce the addendum's table
exactly). The full design now has 43 points (own untied-50k runs at w384-w704 replace Porian's points there).
verify_design.py: ALL PASS.
| design (points) | planted law | median c (M1) | M1 95% set covers c | second term fires | median dBIC |
|---|---|---|---|---|---|
| phase 1 (30) | 1-term, c = 0.1 / 0.5 / 1.0 | 0.10 / 0.50 / 1.00 | 1.00 / 0.96 / 0.92 | 0.00 / 0.045 / 0.01 | -6.8 / -6.6 / -6.4 |
| phase 1 (30) | DM-3 2-term, c = 1.07 | 1.05 (M2) | - | 1.00 | 59 |
| phase 1+2+Porian (43) | 1-term, c = 0.1 / 0.5 / 1.0 | 0.10 / 0.50 / 1.00 | 1.00 / 0.91 / 0.85 | 0.045 / 0.04 / 0.015 | -7.5 / -6.2 / -5.8 |
| phase 1+2+Porian (43) | DM-3 2-term, c = 1.07 | 1.05 (M2) | - | 1.00 | 159 |

## DM-4 Amendment 2 (2026-10-21, after the G1 trainer-gap diagnostic, before any G1 rerun)
**Why.** G1 failed (PHASE0.md): our untied-50k runs sit ~+0.096 (w128) / ~+0.113 (w160) nats behind Porian once the
validation-slice offset (our slice is 0.051 / 0.047 nats easier, porian_check.json) is removed. The diagnostic
(diag_g1.json, job 48811183, w128, full validation slice of 5,878,508 targets, both arms start from Porian's step-1211
weights with a fresh AdamW and train to step 2421 with the same hyperparameters): arm A (our stream, one batch = 28
consecutive windows) 4.8104; arm B (the same windows, globally permuted over the 1,210 steps) 4.7659; Porian's own
step-2421 checkpoint 4.7596. Arm A is +0.0507 behind Porian over those 1,210 steps, arm B +0.0063: batch composition
(consecutive windows of one long document sharing a batch) accounts for 0.0445 nats (88%) of the gap over that stretch,
and 0.006 is left. Adam epsilon, grad clip and init were not varied (the diagnostic starts from Porian's weights);
they remain candidate suspects for the rest only if the rerun still fails.
**What changes.** train.py gets ChunkShuffled: batch s draws its 28 windows from a seeded permutation of the windows of
its 1,024-step chunk (--shuffle-chunk-steps 1024, default on for every run; 0 restores the old order). The windows,
bytes and D_bytes are exactly those of the old stream (only their order inside a chunk changes), the permutation depends
only on (seed, chunk index) so the const leg, cooldown legs and resumed runs agree, and arms stay paired by seed.
Everything else (model, optimiser, LR, budget, runs.csv, evaluation) is unchanged. The diagnostic legs (--end-step)
keep their own order. The fix is applied to all phase 1 and phase 2 runs alike.
**G1 rerun (FarmShare L40S, farmshare/g1.sbatch, results in $DM4/g1r).** untied-50k NeoX, w128 and w160, seeds 0-2, constant LR to D/N = 20,
the same runs.csv rows as the first G1. Gate (rule of PHASE0.md, offset removed): the seed mean of y at D/N = 20 must be
within +-0.05 nats of Porian's 4.796 (w128) / 4.598 (w160) on our slice, i.e. after adding the slice offset 0.051 /
0.047: raw seed mean in [4.695, 4.795] (w128) and [4.501, 4.601] (w160). Both widths must pass. A run landing more than
0.05 BELOW the bar is also a fail (it would mean something other than Porian's recipe).
**If G1 passes:** phase 1 (runs.csv phase 1, ~68 L40S GPU-h) is launched on FarmShare as a resumable array, at most 4
GPUs. **If it fails:** no phase 1; the numbers go to Brando with the next suspect to test (Adam eps, grad clip, init,
warmup start).
