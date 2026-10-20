# DM-4 phase 0 results (Sherlock, Oct 20 2026)

Jobs: dm4-prep 48657441, dm4-prepfull 48658005, dm4-probe 48658006 (TIMEOUT at 45 min after w640), dm4-g1 48658007
(6 tasks), dm4-p0post 48658008. Files here: bpt.json (bytes per token), probe.csv (27 configs, one L40S, eager),
phase0_summary.json (G1 rows). The probe rerun for w704 + compiled repeats + Porian's checkpoints on our full
validation slice is job 48681732 (dm4-probe2).

## G1 (untied-50k, seed mean at D/N = 20, +-0.05 nats of Porian rw tuned-short-const)
| width | seeds (nats / NeoX token) | mean | sd | Porian | diff | verdict |
|---|---|---|---|---|---|---|
| 128 | 4.8798 / 4.8247 / 4.8195 | 4.8413 | 0.033 | 4.796 | +0.045 | pass |
| 160 | 4.6647 / 4.6558 / 4.6704 | 4.6636 | 0.007 | 4.598 | +0.066 | FAIL (~4 SE) |

G1 = FAIL. As pre-registered: no phase 1 until the pipeline is fixed (or a gate change is written down as Amendment 2).

## Is it the validation slice or the trainer?
- Porian's own w128 (step 2421) and w160 (step 3241) checkpoints, scored on the first 120 windows (245,760 tokens) of
  our validation slice during the CPU smoke: 4.768 and 4.558, against Porian's logged val/loss 4.811 and 4.576 at the
  same steps. Our slice is, if anything, slightly easier (-0.04 / -0.02), so the validation-set offset does not explain
  the miss. The full-slice number comes with dm4-probe2 (porian_check.json).
- Same data, model at equal tokens: Porian's curve on our slice at our step counts is ~4.74 (w128, step 2621) and
  ~4.58 (w160, step 3002) (log-step interpolation of its checkpoints). Ours: 4.84 and 4.66. Trainer gap ~+0.08 to
  +0.10 nats at both widths.
- Checked identical to Porian's run records (experiment_results.pickle, rw/tuned/short/const): lr 0.0115/0.0105,
  global bs 28/32 x 2048 (57,344 / 65,536 tokens per step, matching the checkpoints' FLOP tags), beta1 0.9, beta2 0.99,
  warmup 7,503,872 / 9,809,920 tokens (130 / 149 steps; Porian's logged lr 0.0089 at step 100 = ours), independent
  wd 1e-4, z-loss 1e-4, grad clip, amp_bfloat16, qk-norm, seq 2048, vocab 50,432; OpenLM init (trunc-normal 1/sqrt(fan_in),
  depth-scaled output projections, embeddings and head 1/sqrt(w)); weight decay on every parameter as OpenLM.
  The model code loads Porian's checkpoints strictly and reproduces their loss, so the forward pass is not it.
- What is left to differ: the data stream (OpenLM's tokenize-shuffle sequences vs our shard-ordered, in-shard-shuffled
  documents cut into consecutive windows, one batch = 28 consecutive windows), Adam epsilon (OpenLM default 1e-8 assumed),
  grad-clip value (Porian's records say only "True"), and seed spread (w128 sd 0.033, seed 0 alone +0.04).
- Next diagnostic (cheap, decisive for the trainer): start our trainer from Porian's w128 step-1211 weights (fresh Adam)
  and train to step 2421 on our stream; compare with Porian's step-2421 checkpoint on the same slice (~10 L40S-minutes).
  A second arm with globally shuffled windows tests the batch-composition hypothesis.

## Throughput and budget (measured)
One L40S, eager mode: tokens/s from 590k (w128, 8k vocab) to 55k (w640, NeoX). The measured cost of a G1 run matches
the prediction (w128 seed 2: 1,302 s logged vs 1,307 s predicted). budget.md: 166.8 GPU-h in total (phase 1 67.8,
phase 2 99.0), 1.8x the pessimistic FLOP estimate (93.3): small bodies run at low utilisation, the 50k unembedding
dominates. w704 is scaled from w640 by FLOPs per token until dm4-probe2 measures it (46.2 GPU-h of the total).
Phase 1 at 4 FarmShare L40S is ~17 h of wall time. Compiled-mode speedup: from dm4-probe2.
