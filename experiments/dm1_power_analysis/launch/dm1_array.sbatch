#!/bin/bash
# DM-1 on Sherlock as one job array: submit once, Slurm runs every shard (no new login needed).
#   cd $SCRATCH/dark-matter2/experiments/dm1_power_analysis && sbatch launch/dm1_array.sbatch
# Then, after all shards finish: python analyze.py --results results && python verify.py --results results
#SBATCH --job-name=dm1_power
#SBATCH --partition=normal
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dm1_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
set -euo pipefail
mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${VENV:-$SCRATCH/dm_venv}/bin/python   # venv with numpy + scipy (python -m venv $SCRATCH/dm_venv; pip install numpy scipy)
$PY dm1.py --out results --shard "$SLURM_ARRAY_TASK_ID" --nshards 20 --workers "$SLURM_CPUS_PER_TASK"
