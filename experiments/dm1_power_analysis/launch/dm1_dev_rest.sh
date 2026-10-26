#!/bin/bash
# DM-1, windows [1e6,1e9] and [1e5,1e9] (cells 162-323) on Sherlock's dev partition, in the foreground:
#   cd $SCRATCH/dark-matter2/experiments/dm1_power_analysis
#   srun -p dev -c 8 --mem=8G -t 02:00:00 --mail-type=END,FAIL bash launch/dm1_dev_rest.sh > logs/dm1_rest.log 2>&1
# Windows [1e5,1e7] and [1e7,1e9] (cells 0-161) already ran locally; replicates are seeded by (cell, rep), so any
# replicate run in both places is identical and the merge keeps one copy.
set -euo pipefail
mkdir -p logs results_sherlock
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${VENV:-$SCRATCH/dm_venv}/bin/python   # module load python/3.12.1; python3 -m venv $SCRATCH/dm_venv; $SCRATCH/dm_venv/bin/pip install numpy scipy
$PY dm1.py --out results_sherlock --min-cell 162 --workers "${SLURM_CPUS_PER_TASK:-8}"
