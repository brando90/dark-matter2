# DM-1b results (Oct 19, 2026): calibrated likelihood-ratio test in place of the Wald arm

43,200 replicates (324 cells, DM-1's seeds and data), run on Sherlock dev (job 48518244, 4 CPUs, 2,092 s; 5.98M
bootstrap refits after the exact shortcuts). All pre-registered checks pass
(`python verify_dm1b.py --results results_dm1b`: ALL PASS). Tables: `python analyze_dm1b.py --results results_dm1b
--dm1-results results`. Test and predictions: PREREG.md, "DM-1b addendum" (committed in 788fb31 before any run).

## Takeaway
A parametric-bootstrap LRT keeps BIC's power and fixes its calibration: it detects second terms 3-10x weaker than
DM-1's primary rule with a null rate of 0.02-0.07 per ladder (BIC alone: up to 0.117). It does not rescue a
Pythia-scale ladder: on [1e7, 1e9] nothing detects s <= 0.3 with power 0.8, so the limit there is the data, not the test.
Use the LRT as DM-2/DM-3's detection rule.

## Pre-registered checks
| Check | Result |
|---|---|
| Completeness, finite fields, nesting (RSS_2 <= RSS_1), determinism | 43,200 / 43,200 unique; 0 violations; max diff 2.7e-13 |
| Positive control ([1e5,1e9], 8/dec, sd 0.005, s 1, beta 1) | power 1.000, median beta error 0.0012 |
| Worst null cell <= 0.10 | 0.070 ([1e7,1e9], 4/dec, sd 0.02) |
| Pooled null rate <= 0.05 | 0.0488 (12,000 null replicates, MC SE 0.002) |

Fitter agreement with DM-1's stored fits (36,000 testable replicates): RSS_1 identical (1e-6) on 100%; RSS_2 within
0.1% on 93.5%, > 1% worse on 0.22%, > 0.1% better on 4.5%.

## Main readout: smallest s with power >= 0.8 (sd 0.01 nats, 4 points/decade)
| Window | beta | LRT | primary (DM-1) | BIC alone |
|---|---|---|---|---|
| [1e5, 1e7] | 0.68 | 0.1 | none <= 1 | 0.1 |
| [1e5, 1e7] | 1.0 | 0.03 | 0.3 | 0.03 |
| [1e7, 1e9] | 0.68 | none <= 1 | none <= 1 | none <= 1 |
| [1e7, 1e9] | 1.0 | none <= 1 | none <= 1 | none <= 1 |
| [1e6, 1e9] | 0.68 | 0.3 | none <= 1 | 0.3 |
| [1e6, 1e9] | 1.0 | 0.1 | 0.3 | 0.1 |
| [1e5, 1e9] | 0.68 | 0.03 | 0.3 | 0.03 |
| [1e5, 1e9] | 1.0 | 0.03 | 0.1 | 0.03 |

Null rejection rate, pooled / worst testable cell: LRT 0.049 / 0.070, BIC alone 0.040 / 0.117, primary 0.008 / 0.035.

Power at s = 0.1 (sd 0.01, 4/dec), LRT | primary | BIC, and the median |100x extrapolation error| (nats) of the
selected model:
| Window | beta | power LRT / primary / BIC | error LRT-selected / primary-selected / 1-term |
|---|---|---|---|
| [1e5, 1e7] | 0.68 | 0.86 / 0.00 / 0.88 | 0.122 / 0.178 / 0.178 |
| [1e5, 1e7] | 1.0 | 1.00 / 0.25 / 1.00 | 0.103 / 0.649 / 0.654 |
| [1e7, 1e9] | 0.68 | 0.02 / 0.01 / 0.02 | 0.019 / 0.019 / 0.018 |
| [1e7, 1e9] | 1.0 | 0.05 / 0.01 / 0.06 | 0.020 / 0.019 / 0.019 |
| [1e6, 1e9] | 0.68 | 0.45 / 0.07 / 0.32 | 0.029 / 0.031 / 0.030 |
| [1e6, 1e9] | 1.0 | 0.97 / 0.17 / 0.97 | 0.029 / 0.063 / 0.062 |
| [1e5, 1e9] | 0.68 | 1.00 / 0.03 / 1.00 | 0.015 / 0.094 / 0.094 |
| [1e5, 1e9] | 1.0 | 1.00 / 1.00 / 1.00 | 0.011 / 0.011 / 0.309 |

The full LRT power grid (all 324 cells) is in `results_dm1b/cells_dm1b.csv`. Ladders with <= 6 points (2/decade on
the two-decade windows) stay untestable, as in DM-1.

## Predictions
1. **Held.** LRT null rate 0.020-0.070 in every testable cell (BIC alone up to 0.117).
2. **Held.** LRT power tracks BIC alone (same min s in every readout row; power within -0.07 to +0.13 of BIC's at
   s 0.03-1, higher where the ladder is coarse, e.g. 0.45 vs 0.32 at [1e6,1e9], beta 0.68, s 0.1) and the min detectable s drops >= 3x vs the primary rule in three of four windows
   ([1e5,1e7], [1e6,1e9], [1e5,1e9]).
3. **Held.** [1e7, 1e9], 4/dec, sd 0.01: LRT power at s <= 0.1 is 0.02-0.08.

## What it means for the project
- Detection rule for DM-2 and DM-3: the bootstrap LRT. It costs ~200 refits per ladder, trivial for one real ladder.
- Where the second term matters for prediction (ladders down to 1e5, where the 1-term fit is off by 0.1-0.65 nats at
  100x), the LRT finds it and the selected model's extrapolation error falls 2-6x. On [1e7, 1e9] the term is
  invisible but also nearly harmless (~0.02 nats at 100x).
- DM-2's Porian 2nd term (z 3.2/4.3, dBIC ~30) is far past this test's threshold for a 5M-455M ladder, so DM-2's
  open question stays the embedding accounting, not the test.

## Next
Size DM-3 (the tiny CPU ladder) with the LRT: how many sizes per decade, down to which N, and how many seeds for a
given s; re-test DM-2's Porian ladders with the LRT under non-embedding and profiled-embedding parameter counts.
