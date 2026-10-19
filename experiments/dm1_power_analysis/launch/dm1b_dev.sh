#!/bin/bash
# DM-1b (parametric-bootstrap LRT on DM-1's replicates) on Sherlock's dev partition, in the foreground (dev refuses
# sbatch; its QOS caps 4 CPUs and 2 h):
#   cd $SCRATCH/dark-matter2/experiments/dm1_power_analysis
#   srun -p dev -c 4 --mem=16G -t 02:00:00 --mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com \
#       bash launch/dm1b_dev.sh > logs/dm1b.log 2>&1
# Resumable: a second run skips the (cell, rep) already written in results_dm1b/. Arguments are passed to dm1b.py.
set -euo pipefail
mkdir -p logs results_dm1b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${VENV:-$SCRATCH/tdc_gpu/venv_exp19}/bin/python   # any venv with numpy + scipy (python/3.12.1)
$PY dm1b.py --out results_dm1b --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
echo "SRUN_RC=$?" > logs/dm1b_rc.txt
