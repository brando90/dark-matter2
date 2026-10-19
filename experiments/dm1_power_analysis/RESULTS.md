# DM-1 results (Oct 19, 2026)

43,200 replicates (324 cells), all pre-registered checks pass (`python verify.py --results results`: ALL PASS).
Tables: `python analyze.py --results results`. Changes after the first run are in PREREG.md.

## Takeaway
A second power-law term is only visible when the ladder reaches small models. With a Pythia-scale ladder
(1e7-1e9 params) no test sees a term up to the size of the main term at 1M params, and the 1-term fit then
extrapolates 100x out with only ~0.02 nats of error, so the second term barely matters for prediction up there. Down at
1e5 params it is easy to see. The primary rule's Wald arm, not the data, limits detection: BIC alone finds terms
3-10x weaker at a similar false-positive rate.

## Pre-registered checks
| check | result |
|---|---|
| positive control ([1e5,1e9], 8/dec, sd 0.005, s=1, beta=1) | power 1.000, median beta rel. error 0.0012 |
| false positives (primary rule) | worst null cell 0.035, pooled 0.0080 |
| prediction 4: [1e7,1e9] cannot see s <= 0.1 at sd 0.01 | held: power 0.00-0.01 |

## Main readout: smallest s with power >= 0.8 (sd 0.01 nats, 4 points/decade)
s = B N_ref^-beta / (A N_ref^-alpha) at N_ref = 1M params; alpha = 0.34.

| window | beta = 0.68 | beta = 1.0 | 1-term extrapolation error at 100x N_max, s = 0.1, beta = 1 |
|---|---|---|---|
| [1e5, 1e7] tiny CPU ladder | none <= 1 | 0.3 | 0.65 nats |
| [1e7, 1e9] Pythia-like | none <= 1 | none <= 1 | 0.019 |
| [1e6, 1e9] Chinchilla-like | none <= 1 | 0.3 | 0.062 |
| [1e5, 1e9] four decades | 0.3 | 0.1 | 0.31 (selected model: 0.011) |

## Not pre-registered: the Wald arm is the bottleneck
Same cells, BIC alone (dBIC > 2): null FPR pooled 0.040, worst cell 0.117 (Wald alone: pooled 0.110, worst 0.245).

| window, sd 0.01, 4/dec | beta | BIC alone, s = 0.03 / 0.1 / 0.3 / 1 | primary rule |
|---|---|---|---|
| [1e5, 1e7] | 0.68 | 0.26 / 0.88 / 1.00 / 1.00 | 0.00 / 0.00 / 0.12 / 0.47 |
| [1e5, 1e7] | 1.0 | 1.00 / 1.00 / 1.00 / 1.00 | 0.03 / 0.25 / 0.99 / 1.00 |
| [1e7, 1e9] | 0.68 | 0.05 / 0.02 / 0.12 / 0.32 | 0.00 / 0.01 / 0.01 / 0.11 |
| [1e7, 1e9] | 1.0 | 0.09 / 0.06 / 0.16 / 0.50 | 0.00 / 0.01 / 0.01 / 0.24 |
| [1e6, 1e9] | 0.68 | 0.09 / 0.32 / 0.98 / 1.00 | 0.03 / 0.07 / 0.32 / 0.49 |
| [1e6, 1e9] | 1.0 | 0.24 / 0.97 / 1.00 / 1.00 | 0.10 / 0.17 / 0.80 / 1.00 |
| [1e5, 1e9] | 0.68 | 0.85 / 1.00 / 1.00 / 1.00 | 0.16 / 0.03 / 0.92 / 1.00 |
| [1e5, 1e9] | 1.0 | 1.00 / 1.00 / 1.00 / 1.00 | 0.32 / 1.00 / 1.00 / 1.00 |

The 5-parameter fit is sloppy, so the asymptotic Wald SE of B is large even when the fit recovers beta (median beta_hat
0.67-0.74 at s = 0.03-0.1 in [1e5,1e9], beta = 0.68). The primary rule's power is not monotone in s there (0.16 at
s = 0.03, 0.03 at s = 0.1): the few low-s "detections" are wrong-reason fits (median alpha_hat 0.02-0.09 instead of
0.34, E absorbed into the slow term). BIC's worst null cell (0.117) breaks the 0.10 bar, so neither test alone is the
answer.

## Next
Replace the Wald arm with a calibrated likelihood-ratio test (parametric bootstrap of RSS_1 / RSS_2 under the fitted
1-term null), pre-registered as DM-1b, then size the DM-3 tiny CPU ladder with it.
