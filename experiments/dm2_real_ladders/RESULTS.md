# DM-2 results (Oct 19 2026)

**Takeaway.** Only Porian's tuned, constant-LR ladders (rw and owt2, 5M-455M at 20 tokens/param) show a significant
second term, and it is the textbook dark-matter shape: fitted on the sizes >= 100M, the 1-term law under-predicts
every smaller model, by +0.35 to +0.37 nats at 5M. But it is **not yet dark matter**: the same curves show nothing
(and the curvature flips sign) when N counts only non-embedding parameters, and the joint (N, D) law that should
separate it from the data term misfits these runs. No other ladder (Chinchilla 43 sizes, LLaMA 21, Porian base
hparams) shows a second term, and DM-1's power table says those nulls mean "not detectable here", not "absent".

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
