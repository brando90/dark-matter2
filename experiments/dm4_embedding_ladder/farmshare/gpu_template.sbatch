#!/bin/bash
#SBATCH -J dm4-gpu
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/%x-%A_%a.log
# DM-4 GPU job template for FarmShare (L40S). Runs the command given after the script in $CODE with the DM-4 env.
# QOS gpu allows 4 running jobs and 32 submitted per user across ALL experiments: arrays at most %4, and check
# `squeue -r -h -u $USER -q gpu | wc -l` first. One mail per array (never ARRAY_TASKS).
# Example (G1 run 0, or an array whose command reads $SLURM_ARRAY_TASK_ID):
#   mkdir -p ~/dm4/logs && cd ~/dm4 && sbatch -J dm4-g1 -t 01:00:00 code/experiments/dm4_embedding_ladder/farmshare/gpu_template.sbatch \
#     python train.py --root ~/dm4 --index 0 --out ~/dm4/g1/run0 --no-cooldown --log-every 200
set -euo pipefail
source "${DM4_ROOT:-$HOME/dm4}/code/experiments/dm4_embedding_ladder/farmshare/env.sh"   # spool copy: fixed path
[ -f "$VENV/.ready" ] || { echo "no venv: run farmshare/prep.sbatch (or build_venv.sh) first"; exit 1; }
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ $# -gt 0 ] || { echo "usage: sbatch gpu_template.sbatch <command ...>"; exit 1; }
"$@"
echo DM4_GPU_DONE
