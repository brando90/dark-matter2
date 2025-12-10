# Pythia Data Documentation

## Loss Values Source

**Source**: Cerebras-GPT paper Table 8 and Figure 1 ([arXiv:2304.03208](https://arxiv.org/abs/2304.03208))

**Loss Type**: **Pile test set loss** (cross-entropy in nats/token)

The losses are from Figure 1 of the Cerebras-GPT paper:
> "Figure 1: Pile test set loss given pre-training FLOPs for Cerebras-GPT, GPT-J, GPT-NeoX, and Pythia."

See image: `research_papers/plot_pythia_vs_cerebras_gpt_vs_gpt_j_gpt_neox_pile_loss_vs_train_flops.png`

## Verified Final Checkpoint Loss Values

| Model | N_non_embed | N_total | Pile test loss |
|-------|-------------|---------|----------------|
| 70M   | 18,915,328 | 70,426,624 | 2.504 |
| 160M  | 85,056,000 | 162,322,944 | 2.186 |
| 410M  | 302,311,424 | 405,334,016 | 1.971 |
| 1B    | 805,736,448 | 1,011,781,632 | 1.845 |
| 1.4B  | 1,208,602,624 | 1,414,647,808 | 1.793 |
| 2.8B  | 2,517,652,480 | 2,775,208,960 | 1.720 |
| 6.9B  | 6,444,163,072 | 6,856,253,440 | 1.626 |
| 12B   | 11,327,027,200 | 11,842,140,160 | 1.582 |

## Column Descriptions

| Column | Description | Source |
|--------|-------------|--------|
| N_non_embed_params | Non-embedding parameters | Pythia Paper Table 1 ([arXiv:2304.01373](https://arxiv.org/abs/2304.01373)) |
| N_total_params | Total parameters (non-embed + embeddings) | Calculated: N_non_embed + vocab_size × d_model |
| tokens_D | Training tokens (~300B for final checkpoint) | Pythia Paper |
| loss | Pile test cross-entropy (nats/token) | Cerebras-GPT Table 8 ([arXiv:2304.03208](https://arxiv.org/abs/2304.03208)) |
| flops_C | Training FLOPs = 6 × N_non_embed × D | Calculated using non-embedding params |
| family | Model family identifier | "pythia" |
| loss_comment | Loss type indicator | "pile_test" = Pile test set loss |

## Parameter Calculation Details

**Non-embedding params**: From Pythia paper Table 1 (verified)

**Pythia uses UNTIED embeddings** (separate input embedding and output projection layers), so embedding params are counted **twice**:

**Embedding params** = 2 × (vocab_size × d_model)
- Pythia vocab_size = 50,304
- d_model varies by model size

**Total params** = N_non_embed + 2 × (vocab_size × d_model)

| Model | d_model | N_embed (×2) | N_total |
|-------|---------|--------------|---------|
| 70M   | 512     | 51,511,296   | 70,426,624 ≈ 70M |
| 160M  | 768     | 77,266,944   | 162,322,944 ≈ 162M |
| 410M  | 1024    | 103,022,592  | 405,334,016 ≈ 405M |
| 1B    | 2048    | 206,045,184  | 1,011,781,632 ≈ 1.01B |
| 1.4B  | 2048    | 206,045,184  | 1,414,647,808 ≈ 1.41B |
| 2.8B  | 2560    | 257,556,480  | 2,775,208,960 ≈ 2.78B |
| 6.9B  | 4096    | 412,090,368  | 6,856,253,440 ≈ 6.86B |
| 12B   | 5120    | 515,112,960  | 11,842,140,160 ≈ 11.84B |

## Notes

- All loss values are for the **final checkpoint** (~300B tokens training)
- Loss is **cross-entropy** measured in **nats/token** (natural log units)
- Perplexity = exp(loss), e.g., for 70M: perplexity ≈ e^2.504 ≈ 12.2
- The Cerebras-GPT paper normalized all cross-entropy results to be comparable to GPT-2 vocabulary
- **FLOPs note**: Cerebras-GPT Table 2 reports slightly higher FLOPs (~6.1e21 for 2.8B vs our 4.5e21). This is because we use non-embedding params for the standard C=6ND formula, while Cerebras may use a different calculation.

---

## Appendix: Intermediate Checkpoint Losses NOT Available

After exhaustive search (Pythia paper, GitHub evals, HuggingFace, W&B reports), **NO publicly accessible source contains training loss values for intermediate checkpoints**.

### Sources Checked (All Failed to Provide Intermediate Loss Values):
1. **Pythia Paper (arXiv:2304.01373)** - Does NOT publish loss values at any checkpoint
2. **GitHub evals/pythia-v1/** - Contains downstream task accuracy (ARC, LAMBADA, PIQA, etc.), NOT training loss
3. **HuggingFace model cards** - No loss values stored
4. **W&B reports** - Show loss charts but require authentication to extract exact values
5. **GitHub Issue #194** - https://github.com/EleutherAI/pythia/issues/194 - Still OPEN with no answer

### To Get Intermediate Checkpoint Losses

You would need to either:

1. **Use W&B API with authentication:**
   ```python
   import wandb
   wandb.login()
   api = wandb.Api()
   runs = api.runs("eleutherai/pythia")
   for run in runs:
       history = run.history(keys=["train/lm_loss"])
       # Extract values at specific steps
   ```

2. **Run evaluations on each checkpoint:**
   ```python
   from transformers import GPTNeoXForCausalLM
   model = GPTNeoXForCausalLM.from_pretrained(
       "EleutherAI/pythia-70m",
       revision="step1000"  # Or step10000, etc.
   )
   # Evaluate on pile test set
   ```

3. **Contact EleutherAI:**
   - stella@eleuther.ai
   - hailey@eleuther.ai
