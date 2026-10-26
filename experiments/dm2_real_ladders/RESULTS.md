# DM-3 results (Oct 19 2026): embedding accounting vs a real second term, Porian tuned-short-const rw / owt2

**Takeaway.** The excess loss of the small tuned models is **not embedding accounting**, and it is **not yet shown
to be dark matter**. Pre-registered verdict: **undetermined**. Weighting embeddings by a free factor c
(N_c = N_nonemb + c N_emb) cannot replace the second term. The best 1-term-with-c fit leaves ~2x the residual sd of
the 2-term fit, and dBIC still favours the 2-term law by ~20 with c free on both datasets. But once c is free, the
second term's size and exponent are not identified (bootstrap z 0.01 / 0.09). The Kaplan joint law fits these
curves ~3x better than the Chinchilla form and needs no second N term, but it still misfits at ~6x seed noise.
Post hoc, its data term reproduces only ~15% of the small-model excess. Next: our own ladder that varies embedding
size at fixed body size, with seeds at the small end.

Code: dm3.py (fits, 1,000-refit bootstraps), dm3_summary.py, dm3_posthoc.py (labelled post hoc), verify.py (ALL
PASS, 6 DM-3 checks added). PREREG.md "DM-3 addendum" was committed and pushed (0be63b2) before any DM-3 fit; the
changes after the smoke run are listed there. Full tables: results/dm3_summary.md.

## A3: N-only at D = 20N, c profiled (14 sizes, 5M-455M)
| | rw | owt2 |
|---|---|---|
| best c, 1-term law (c1) | 2.0 (grid edge) | 0.045 (E hits 0) |
| 95% profile set for c1 | [0.05, 0.11] U [1.55, 2.0+] | [0.03, 0.08] U [1.83, 2.0+] |
| c1 bootstrap 2.5-97.5% | 0.09-2.0 | 0.03-1.40 |
| residual sd: 1-term + c / 2-term + c | 0.014 / 0.006 | 0.019 / 0.008 |
| dBIC (1-term + c vs 2-term + c, k 4 vs 6) | +19.5 | +20.8 |
| best c, 2-term law (c2), 95% set | 1.07, [0.52, 2.0+] | 1.67, [0.75, 2.0+] |
| B, beta at c2; Wald z; bootstrap z (c re-profiled per refit) | 3.2, 1.46; 2.75; 0.01 | 10.8, 1.67; 1.40; 0.09 |
| primary rule / shows a second term | yes / no | no / no |

- c is not identified by a 1-term law. Its RSS profile has two separate low regions: c ~ 0.05, where E is
  forced to 0 (a pure power law with alpha 0.09), and c >= 1.6. Post hoc, past the grid, the 1-term RSS bottoms out at
  c ~ 3 (rw) / ~5 (owt2), where an embedding parameter is worth 3-5 body parameters, which is not physical. Even
  there it is still 4x the 2-term RSS (0.0014 vs 0.00033 rw; 0.0024 vs 0.00055 owt2). c = 1 is outside the 95% set on
  both datasets.
- With c fixed anywhere in 0.2-2.0, dBIC favours the 2-term law by 14-30. Its Wald z is > 2 only for c in 0.2-1.2
  (rw) / 0.2-1.4 (owt2), peaking at 10-11 near c = 0.5. Past that, B and beta trade off and z falls to 0.5-0.9 at
  c = 2. Only at c ~ 0.05-0.1 does B drop to 0, and there E = 0.
- Hold-out (fit the smaller 10 sizes, predict the largest 4), RMSE 1-term / 2-term:
  - c from all data: 0.045 / 0.034 (rw) and 0.040 / 0.031 (owt2).
  - Strict (c re-profiled on the 10): 0.045 / 0.019 and 0.072 / 0.023.
  - c = 1 (DM-2): 0.072 / 0.016 and 0.099 / 0.009.
  - The 2-term law predicts the large sizes better in every variant.
- Reverse ladder (1-term fitted on the 5 sizes >= 100M; mean residual of the 9 smaller sizes):
  - At c1: rw +0.062 (9/9 positive, 5M +0.20). owt2 -0.073 (0/9 positive, 5M -0.16; this is the E = 0 solution
    overshooting).
  - c re-fitted on the large sizes (0.68 / 0.65): +0.110 / +0.105, 5M +0.37 / +0.40, the same as DM-2's c = 1.
  - The 2-term fit at c2 leaves every size within +-0.016 nats.

## B3: Kaplan joint law L = E + (A n^-p + G/d)^q [+ B n^-beta] on the constant-LR curves (D/N 2.5-320)
| | rw (93 pts) | owt2 (89 pts) |
|---|---|---|
| residual sd: Chinchilla form / Kaplan K0 (E = 0) / K1 (+E) | 0.089 / 0.035 / 0.031 | 0.082 / 0.042 / 0.035 |
| K1 best c, 95% set | 0.08, [0.04, 0.18] | 1.5 (edge), [0.08, 0.22] U [0.97, 1.5+] |
| K1 sd at c = 0 / c = 1 | 0.033 / 0.032 | 0.039 / 0.036 |
| K1 alpha_N = p q, alpha_D = q | 0.13, 0.21 | 0.32, 0.30 |
| K2 second term: Wald z, dBIC, bootstrap z (c fixed) | 0.10, -2.3, 1.9 | 0.08, -2.7, 1.5 |
| hold-out RMSE K1 / K2 (5 largest of 16 sizes) | 0.031 / 0.045 | 0.060 / 0.049 |
| reverse-ladder mean K1 / K2 (fit on 7 sizes >= 100M) | -0.090 / +0.017 | -0.012 / -0.012 |

- The Kaplan form is the better joint law for constant-LR curves (prediction 4 held). In it, c hardly matters: sd
  changes by < 0.004 between c = 0 and 1. A second N term is not significant at any c (prediction 5 held).
- Its residual sd (0.03) is ~6-18x the seed sd at 15-57M (0.002-0.005). Correlated points from one run make every
  joint SE optimistic. So the joint fit cannot see a term the size of the A3 excess: along D = 20N the K1 surface
  misses the data by RMS 0.021-0.025.
- **Post hoc (dm3_posthoc.py):** along D = 20N, the K1 surface (data term, no fast term) gives a reverse-ladder mean
  of +0.013 / +0.017 (5M +0.05 / +0.07), against +0.096 / +0.107 (5M +0.37 / +0.35) in the data. So the finite-data
  term explains ~15% of the signature, not all of it. At c = 1 it explains none of it (K1 is then nearly a single
  power law along the slice).

## What this says
1. **Embedding accounting: rejected as the whole story.** No embedding weight c in [0, 2], or even c up to 1e4,
   lets a 1-term law fit the 20N curve as well as the 2-term law. The c that comes closest is either unphysical
   (c ~ 3-5) or needs E = 0.
2. **Dark matter: not established.** The extra curvature is real (dBIC ~20 with c free, better hold-out), but its
   shape (B, beta) trades off against c and E. On 14 points, a second term with c free cannot be told apart from
   the many curves that fit equally well. The joint law that should settle it misfits by more than the effect.
3. **The data term at 20 tokens/param is not the answer either** (post hoc: ~15% of the excess).

## Pre-registered predictions (DM-3)
1. c1 < 0.5 with its CI excluding 1: **held for owt2 (0.045), failed for rw (2.0)**. The CI excludes 1 on both.
2. With c free, the second term fails the primary rule on both: **held for owt2, failed for rw** (z 2.75,
   dBIC 19.5). Neither shows a term (bootstrap z 0.01 / 0.09).
3. Reverse-ladder mean at c1 within +-0.05: **failed** (+0.062 / -0.073).
4. K1 fits better than the Chinchilla form: **held** (0.031 / 0.035 vs 0.089 / 0.082).
5. K2's second term not significant: **held**.

## Next step (DM-4)
Separate the count from the curve by design. Train a small ladder (1M-100M) where embedding size varies at fixed
body size: untied vs tied embeddings, and vocab 8k / 32k / 50k at the same width and depth. Use 3+ seeds at each of
the five smallest sizes (Porian's rw 5M seed sd is 0.06) and annealed runs at a few D/N. If loss at a fixed body
size moves with N_emb only by a fitted c < 1, and the excess survives at fixed N_emb, the second term is real.

---

# DM-2 results (Oct 19 2026)

**Takeaway.** Only Porian's tuned, constant-LR ladders (rw and owt2, 5M-455M at 20 tokens/param) show a significant
second term, and it is the textbook dark-matter shape: fitted on the sizes >= 100M, the 1-term law under-predicts
every smaller model, by +0.35 to +0.37 nats at 5M. But it is **not yet dark matter**: the same curves show nothing
(and the curvature flips sign) when N counts only non-embedding parameters, and the joint (N, D) law that should
separate it from the data term misfits these runs. No other ladder (Chinchilla 43 sizes, LLaMA 21, Porian base
hparams) shows a second term, and DM-1's power table says those nulls mean "not detectable here", not "absent".

Sizes at D = 20N (total parameters; a size needs a run at or past 20 tokens/param, and for Chinchilla two runs that bracket it, with tokens/param on Hoffmann's reported N):
- Porian: 14 of 16 on every curve, except owt2 tuned-long-const with 11 of 13 (the data has no 28M, 37M or 57M run for it). The 611M and 901M runs end at about 11 and 5 tokens/param.
- Chinchilla: 28 of 43, kept range 202M-6.3B. Dropped: the nine sizes from 54M to 183M, and also 234M. All of their runs are past 20 tokens/param, with the shortest at 21.5-294, and 54M has only one run. 2,127M and the four sizes of 8.6B or more are dropped too, because all of their runs fall short of 20 (longest 1.9-14.9). 202M, 259M, 2,113M and 2.4B-6.3B have runs on both sides of 20 and are kept.
- LLaMA: 15 of 21. All six sizes of 0.9B and up are dropped: their runs end at 1.4-16 tokens/param.

Full tables: results/summary.md. Code: data.py (ladders), fitlib.py (DM-1 fits + joint fit), dm2.py (fits,
1,000-refit bootstraps), verify.py (ALL PASS), PREREG.md (written before the first fit, with the changes after).

## The numbers
| Ladder (D = 20N, total params) | sizes | Wald z | dBIC | boot z | beta | hold-out RMSE 1-term / 2-term | reverse ladder (mean, positive) |
|---|---|---|---|---|---|---|---|
| Porian rw tuned-short-const | 14 (5M-455M) | 3.23 | 29.1 | 2.68 | 1.42 | 0.072 / 0.016 | +0.107, 9/9 |
| Porian owt2 tuned-short-const | 14 | 4.35 | 29.8 | 2.99 | 1.31 | 0.099 / 0.009 | +0.096, 8/9 |
| Porian rw / owt2 tuned-long-const | 14 / 11 | 0.67 / 0.21 | -1.2 / -3.5 | 0.11 / 0.10 | - | 0.093 / 0.071 (no gain) | +0.110 / -0.007 |
| Porian base hparams (6 curves) | 14 each | <= 0.03 | < 0 | - | - | no gain (2-term worse on kaplan-long) | +0.47 to +0.64 (mis-tuned small models) |
| Chinchilla (Hoffmann, Epoch points) | 28 (202M-6.3B) | 0.16 | -3.8 | 0.01 | - | 0.024 / 0.024 | - (no sizes < 100M at 20N) |
| LLaMA (MAD) | 15 (40M-650M) | 0.01 | -5.3 | 0.02 | - | 0.116 / 0.116 | +0.127, 2/2 |

- Fast term in the two positive curves: 8-12% of the main term at 5M; alpha drops from 0.25-0.30 (1-term) to
  0.13-0.14 with E = 0.6-1.2 nats (the usual E/alpha trade-off; condition number ~1e6). Residual sd of the 2-term fit
  0.006-0.008 nats vs seed sd 0.005-0.062 at 5-10M (20 tok/param) and 0.002-0.005 at 15-57M (200 tok/param).
- Reverse-ladder residuals (rw, 1-term fit on 108M-455M): 85M +0.005, 57M +0.017, 37M +0.031, 22M +0.065,
  15M +0.083, 10M +0.149, 7.5M +0.228, 5M +0.347. owt2: -0.001 ... +0.371. Monotone, growing as N shrinks.
- **H5 (parameter count) decides it.** With non-embedding counts (embeddings are 94% of the 5M model, 8% of 901M),
  every Porian curve has B = 0, alpha falls to 0.07-0.19, and the reverse residuals turn negative (5M: -0.41 / -0.43).
  So the excess loss of small models under total counts is the same thing as embeddings being worth less than body
  parameters. Chinchilla's verdict does not move with the count (total, non-embedding, reported: all null).
- **Analysis B (joint L = E + A N^-a + B N^-b + G D^-g, annealed finals):** B is not significant anywhere (z <= 0.04,
  dBIC -4 to -11). But the law itself misfits: residual sd 0.06 (Chinchilla, SVG-extracted points), 0.21 (Porian
  base-hparam runs), 0.08-0.09 (post hoc, tuned constant-LR curves), against seed sd <= 0.06. A small second term
  cannot be tested inside a misfit that large; LLaMA (28 runs) fits to 0.018 and has no B (z 0.04, boot z 1.8).
- **Final loss per size is not a ladder.** Using each size's last checkpoint (tokens/param from ~1,250 at 5M to ~5 at
  901M in Porian), every curve gives B = 0, and Porian's reverse residuals go wild (-3 to -2,290 nats): the undertrained large
  models bend the curve the other way. This is also why the repo's earlier fit (width x depth as N, all checkpoints
  together) was meaningless.

## Reading the nulls with DM-1
Porian, Chinchilla and LLaMA at 20N span 1.2-1.9 decades (5M-455M, 202M-6.3B, 40M-651M); the nearest DM-1 cell is [1e7, 1e9] at 8 points/decade.
There the primary rule's power is <= 0.03 for s <= 0.3 and 0.04-0.54 at s = 1 (s = fast/main term at 1M params),
with 0 false positives in 400 null replicates. So: a null on these ladders rules out only a very large term; and the
positive Porian result, if real, corresponds to s ~ 0.5-0.8 at 1M (8-12% at 5M, beta - alpha ~ 1.2); DM-1's power at
this noise is between 0.03 (s = 0.3) and 0.33-0.54 (s = 1), so a detection was possible but not likely: if the term
is real, it is big. (DM-1's cells are the partial run, before its Wald fix; the fix lowers
[1e7, 1e9] power slightly, e.g. 0.238 -> 0.210 at s = 1 pooled.)

## Pre-registered predictions
1. No curve passes at 20N: **failed** (Porian tuned-short-const, both datasets).
2. Literal final-loss curves give larger dBIC than 20N: **failed**; they give B = 0 (the confound bends the curve the
   other way).
3. B not significant in >= 3 of 4 joint ladders: **held** (0 of 4), with the misfit caveat above.
4. Porian 20N sub-100M reverse residual within +-0.05: **failed** (+0.10 tuned-short, +0.47 to +0.64 base).
5. Non-embedding counts move alpha by > 0.05 and change a verdict: **held** (alpha 0.25 -> 0.07; two verdicts flip).

## Next step
Settle H5 before anything else: refit the Porian tuned curves with N = N_nonemb + c * N_emb for c in [0, 1] (one
extra parameter, profile c) and with the Kaplan-style joint law, which fits constant-LR curves better than the
Chinchilla form. If the fast term survives at the best-fitting c, it is dark matter; if c alone absorbs it, the "dark
matter" is embedding accounting, and DM-3 (our own tiny ladder) should vary embedding size on purpose.
