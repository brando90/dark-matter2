# DM-4 phase 1, w128 / bpe32k and NeoX-50k rows, run on mercury1 (finished 10-26-2026)

**Status:** DONE, exploratory hardware look. Twelve phase-1 rows (runs.csv array_index 14, 18, 19, 20, 24, 25 for vocab 32,768 own-bpe; 26, 32, 33, 34, 38, 39 for NeoX vocab 50,432): width 128, depth 4, tied vs untied embeddings, seeds 0-2. All 12 ran to rc=0 with `done: true`. Not the frozen L40S launch; PREREG was not edited. With the six 8k rows of 10-25 (results_at1211.md) the grid is complete at width 128 for three vocabularies.

**Takeaway:** tied vs untied at w128, 20 tokens per parameter, const schedule, validation loss y (nats per target token), n=3 seeds per cell, no interval or test run. The sign of untied minus tied changes with vocabulary: 8k untied lower in all three paired seeds (-0.0187), 32k untied lower on average but only two of three seeds (-0.0033, seed 1 is +0.0168), 50k untied higher on average (+0.0088; two of three seeds positive). The 32k and 50k gaps are within about one seed sd (sd 0.008-0.013), so they are not distinguishable from zero here; only the 8k gap (about 3 seed sds) stands out. y is per token, so absolute y is not comparable across vocabularies; nats per byte is comparable and gives the same signs: 8k -0.0045, 32k -0.0008, 50k +0.0021.

| vocab | tied y@20 mean (sd) | untied y@20 mean (sd) | paired untied minus tied y (seeds 0,1,2) | tied nats/byte | untied nats/byte | paired nats/byte diff |
|---|---|---|---|---|---|---|
| bpe8k (8,192, own-bpe; Oct 24) | 4.9791 (0.0038) | 4.9603 (0.0068) | -0.0187 (-0.0140, -0.0265, -0.0157) | 1.1709 | 1.1664 | -0.0045 |
| bpe32k (32,768, own-bpe) | 4.7445 (0.0103) | 4.7412 (0.0078) | -0.0033 (-0.0153, +0.0168, -0.0112) | 1.1157 | 1.1149 | -0.0008 |
| NeoX 50k (50,432) | 4.7253 (0.0022) | 4.7340 (0.0126) | +0.0088 (+0.0159, +0.0137, -0.0033) | 1.1112 | 1.1132 | +0.0021 |

y = validation negative log-likelihood per target token on the full val slice, read at the 2.5/5/10/20 TPP checkpoints of one const-lr run (same y as the G1 gate and results_at1211.md).

| vocab | run | embeddings | seed | y 2.5 TPP | y 5 | y 10 | y 20 | nats/byte @20 | GPU min |

| bpe32k (32,768, own-bpe) | 14 | tied | 0 | 5.7685 | 5.3952 | 4.9436 | 4.7536 | 1.1178 | 39 |
| bpe32k (32,768, own-bpe) | 18 | tied | 1 | 5.8181 | 5.4078 | 4.9580 | 4.7333 | 1.1131 | 29 |
| bpe32k (32,768, own-bpe) | 19 | tied | 2 | 5.7554 | 5.3966 | 5.0436 | 4.7465 | 1.1162 | 29 |
| bpe32k (32,768, own-bpe) | 20 | untied | 0 | 5.7799 | 5.4077 | 4.9634 | 4.7383 | 1.1142 | 39 |
| bpe32k (32,768, own-bpe) | 24 | untied | 1 | 5.7753 | 5.3789 | 4.9758 | 4.7501 | 1.1170 | 29 |
| bpe32k (32,768, own-bpe) | 25 | untied | 2 | 5.7622 | 5.3623 | 4.9467 | 4.7353 | 1.1135 | 29 |
| NeoX 50k (50,432) | 26 | tied | 0 | 5.7627 | 5.3477 | 4.9128 | 4.7260 | 1.1113 | 57 |
| NeoX 50k (50,432) | 32 | tied | 1 | 5.7562 | 5.3527 | 4.9155 | 4.7270 | 1.1116 | 42 |
| NeoX 50k (50,432) | 33 | tied | 2 | 5.7236 | 5.3103 | 4.9098 | 4.7228 | 1.1106 | 42 |
| NeoX 50k (50,432) | 34 | untied | 0 | 5.7408 | 5.3830 | 5.0284 | 4.7419 | 1.1151 | 57 |
| NeoX 50k (50,432) | 38 | untied | 1 | 5.7523 | 5.3834 | 4.9906 | 4.7407 | 1.1148 | 42 |
| NeoX 50k (50,432) | 39 | untied | 2 | 5.7246 | 5.3547 | 4.9356 | 4.7195 | 1.1098 | 42 |

## Settings (from each result.json meta.args)
- Host mercury1.stanford.edu, GPUs 7 (32k rows; also did the 32k data prep first) and 8 (50k rows), RTX A4000 16 GB (frozen plan is L40S, wall-clock not comparable), torch 2.8.0+cu128, no torch.compile, no cooldown (const only), one run at a time per GPU.
- Model: width 128, depth 4, n_nonemb_ref 1,048,576. Optimizer: lr 0.0115, beta2 0.99, independent wd 1e-4, batch size 28, warmup 7,503,872 tokens, 20 tokens per parameter, evals at [2.5, 5, 10, 20] TPP. Steps: 2,643-2,647 (32k), 2,618 (50k).
- Raw per-run results: experiments/dm4_embedding_ladder/results_at1336/run<index>_result.json.
