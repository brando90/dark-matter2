#!/bin/bash
#SBATCH -J dm4-diag-g1
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/%x-%j.log
# G1 trainer-gap diagnostic (AT-669; one GPU, about 25 min on an L40S: 2 arms x 1,210 steps x 0.48 s + 4 full-slice evals).
# Downloads Porian's w128 step-1211 and step-2421 checkpoints (cached in $DM4/porian_ckpts), trains arm A (our stream)
# and arm B (--shuffle-windows) from step 1211 to 2421 at w128 untied-50k (lr 0.0115, bs 28, beta2 0.99, fresh AdamW),
# evaluates both plus Porian's two checkpoints on the FULL validation slice. Output: $DM4/g1diag/diag_g1.json
# (+ armA/, armB/ with result.json and train.log). Phase 1 is not launched by this job.
# FarmShare twin (L40S). QOS gpu: 4 running / 32 submitted per user across all experiments (check squeue first).
# Submit: mkdir -p ~/dm4/logs && cd ~/dm4 && git -C code pull && sbatch code/experiments/dm4_embedding_ladder/farmshare/diag_g1.sbatch
set -euo pipefail
source "${DM4_ROOT:-$HOME/dm4}/code/experiments/dm4_embedding_ladder/farmshare/env.sh"   # spool copy: fixed path
[ -f "$VENV/.ready" ] || { echo "no venv: run farmshare/prep.sbatch first"; exit 1; }
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python diag_g1.py --root "$DM4" --out "$DM4/g1diag" --log-every 50
echo G1_DIAG_DONE
