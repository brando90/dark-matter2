#!/bin/bash
#SBATCH -J dm4-diag-g1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C "GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/diag-g1-%j.log
# G1 trainer-gap diagnostic (AT-669; one GPU, about 25 min on an L40S: 2 arms x 1,210 steps x 0.48 s + 4 full-slice evals).
# Downloads Porian's w128 step-1211 and step-2421 checkpoints (cached in $DM4/porian_ckpts), trains arm A (our stream)
# and arm B (--shuffle-windows) from step 1211 to 2421 at w128 untied-50k (lr 0.0115, bs 28, beta2 0.99, fresh AdamW),
# evaluates both plus Porian's two checkpoints on the FULL validation slice. Output: $DM4/g1diag/diag_g1.json
# (+ armA/, armB/ with result.json and train.log). Phase 1 is not launched by this job.
# Any bf16-capable GPU (compute capability 8.x/9.x: A100, A40, A6000, L40S, H100), never V100/P100. If Slurm rejects the
# -C list, use -C "GPU_SKU:L40S|GPU_SKU:H100_SXM5|GPU_SKU:A100_SXM4" on the command line.
# Submit from $DM4 (mkdir -p logs): CODE=$DM4/code/experiments/dm4_embedding_ladder sbatch --export=ALL $CODE/sherlock/diag_g1.sbatch
set -euo pipefail
source "$CODE/sherlock/env.sh"   # sbatch runs a spool copy: $CODE comes from the submit env
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python diag_g1.py --root "$DM4" --out "$DM4/g1diag" --log-every 50
echo G1_DIAG_DONE
