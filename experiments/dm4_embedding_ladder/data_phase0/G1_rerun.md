# G1 rerun (PREREG Amendment 2), FarmShare L40S, array 1863129 (dm4-g1r), code 2aafa48

untied-50k NeoX, constant LR to D/N = 20, windows shuffled within 1,024-step chunks. All 6 tasks COMPLETED (exit 0:0), 2.53 GPU-h.
y = validation loss (nats/token) on our slice. Raw seed means (3 seeds each).

| width | D/N | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|---|
| 128 | 2.5 | 5.7379 | 5.7533 | 5.7227 | 5.7379 | 0.0153 |
| 128 | 5 | 5.3508 | 5.3950 | 5.3662 | 5.3707 | 0.0224 |
| 128 | 10 | 4.9510 | 5.0068 | 5.0662 | 5.0080 | 0.0576 |
| 128 | 20 | 4.7399 | 4.7271 | 4.7578 | **4.7416** | 0.0154 |
| 160 | 2.5 | 5.5814 | 5.6263 | 5.5532 | 5.5870 | 0.0369 |
| 160 | 5 | 5.1485 | 5.1965 | 5.1247 | 5.1566 | 0.0366 |
| 160 | 10 | 4.7407 | 4.7528 | 4.7331 | 4.7422 | 0.0100 |
| 160 | 20 | 4.5517 | 4.5537 | 4.5560 | **4.5538** | 0.0022 |

Seed order within a row is runs.csv seed 0, 1, 2 (w128 array indices 34, 38, 39; w160 indices 71, 75, 76).

## Gate (Amendment 2): raw seed mean at D/N = 20 in [4.695, 4.795] (w128) and [4.501, 4.601] (w160)
- w128: 4.7416 in [4.695, 4.795]: PASS (Porian 4.796 minus slice offset 0.051 = 4.745; diff -0.003)
- w160: 4.5538 in [4.501, 4.601]: PASS (Porian 4.598 minus 0.047 = 4.551; diff +0.003)

**G1 PASS (both widths). Phase 1 is launched on FarmShare.**
