# DM-4 phase 1, w128 / bpe8k rows, run on mercury1 (AT-1211 weekend run, finished 10-25-2026 00:02 PDT)

**Status:** DONE, exploratory hardware look. Six phase-1 rows (runs.csv array_index 0, 6, 7, 8, 12, 13): width 128, depth 4, vocab bpe8k, tied vs untied embeddings, seeds 0-2. Not the frozen L40S launch; the protocol in PREREG (incl. Amendment 2 chunk-shuffled windows) was not edited.

**Takeaway:** at width 128 / 8k vocab and 20 tokens per parameter, untied is lower than tied on the const schedule in all three paired seeds: y(const, 20 TPP) tied mean 4.9791 (sd 0.0038), untied mean 4.9603 (sd 0.0068); paired untied minus tied = -0.0187 (per seed -0.0140, -0.0265, -0.0157). n=3, no interval or test run here; the gap is about 3 seed sds but is a single-width, single-vocab look. Cooldown (seed 0 only, as in runs.csv) at 20 TPP: tied 4.8671, untied 4.8471.

y = validation negative log-likelihood per target token on the full val slice (the same y as the G1 gate), read at the 2.5/5/10/20 TPP checkpoints of one const-lr run.

| run | embeddings | seed | y const 2.5 TPP | y const 5 | y const 10 | y const 20 | y cool@20 | GPU min |
|---|---|---|---|---|---|---|---|---|
| 0 | tied | 0 | 6.0389 | 5.5688 | 5.1764 | 4.9817 | 4.8671 | 43 |
| 6 | tied | 1 | 6.0197 | 5.6026 | 5.1396 | 4.9809 | not run | 31 |
| 7 | tied | 2 | 6.0454 | 5.5042 | 5.1339 | 4.9747 | not run | 31 |
| 8 | untied | 0 | 6.0096 | 5.5136 | 5.1435 | 4.9677 | 4.8471 | 43 |
| 12 | untied | 1 | 6.0053 | 5.6439 | 5.1300 | 4.9544 | not run | 31 |
| 13 | untied | 2 | 6.0247 | 5.6620 | 5.1654 | 4.9590 | not run | 31 |

## Settings (recorded at run time, from each result.json meta.args)
- Host mercury1.stanford.edu, GPU 9 only (RTX A4000 16 GB; the frozen plan is L40S, so wall-clock is not comparable), venv torch 2.8.0+cu128, no torch.compile, one run at a time.
- Model: width 128, depth 4, n_nonemb_ref 1,048,576; vocab bpe8k (train.py reads bpt['neox'] from the prep stats).
- Optimizer: lr 0.0115, beta2 0.99, independent wd 1e-4, batch size 28, warmup 7,503,872 tokens, 3,112 steps (3,111 for seed 2), 20 tokens per parameter, evals at [2.5, 5, 10, 20] TPP, cooldown at [5, 10, 20] TPP only for array_index 0 and 8.
- Raw per-run results: experiments/dm4_embedding_ladder/results_at1211/run<index>_result.json.
- A first launch at 20:24 PDT crashed within 2 s per run (rc=1, missing neox tokenizer stats); data prep was redone and all six ran to rc=0 from 20:31 PDT (about 31 min each, 43 min for the two with cooldown).
