#!/bin/bash
#SBATCH -J dm4-p1
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-91%4
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/%x-%A_%a.log
# DM-4 phase 1 (runs.csv phase 1: 92 const rows, each with its WSD cooldown branches; ~68 L40S GPU-h, longest run ~1.7 h)
# on FarmShare. Launch ONLY after G1 passes (PREREG Amendment 2). QOS gpu: 4 running / 32 submitted per user across all
# experiments, so the array is %4 and tasks are submitted in waves if the queue holds other jobs (override with
# --array=0-31%4 etc.). Resumable: train.py checkpoints every 30 min to $OUT/ckpt.pt and result.json, --requeue restarts
# from there, and a finished run ("already complete") is a no-op, so the whole array can simply be resubmitted.
# Submit: mkdir -p ~/dm4/logs && cd ~/dm4 && git -C code pull && sbatch code/experiments/dm4_embedding_ladder/farmshare/phase1.sbatch
set -euo pipefail
source "${DM4_ROOT:-$HOME/dm4}/code/experiments/dm4_embedding_ladder/farmshare/env.sh"   # spool copy: fixed path
[ -f "$VENV/.ready" ] || { echo "no venv: run farmshare/prep.sbatch first"; exit 1; }
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
idx=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open("runs.csv")) if r["phase"] == "1" and r["run"] == "const"]
assert len(rows) == 92, len(rows)
print(rows[int(sys.argv[1])]["array_index"])
PY
)
python train.py --root "$DM4" --index "$idx" --out "$DM4/phase1/run$idx" --log-every 200
echo P1_RUN_DONE "$idx"
