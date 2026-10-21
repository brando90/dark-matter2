#!/bin/bash
#SBATCH -J dm4-g1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_SKU:L40S
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/g1-%A_%a.log
# Anchor gate G1: untied-50k NeoX (Porian's exact config) at w128 and w160, seeds 0-2, constant LR to D/N = 20,
# evaluated at D/N = 2.5, 5, 10, 20. Constant-LR leg only (the cooldown branches belong to phase 1).
set -euo pipefail
source "$CODE/sherlock/env.sh"   # sbatch runs a spool copy: $CODE comes from the submit env
cd "$CODE"
idx=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open("runs.csv")) if r["width"] in ("128", "160") and r["vocab"] == "50432"
        and r["tie"] == "untied" and r["tokenizer"] == "neox" and r["run"] == "const"]
rows.sort(key=lambda r: (int(r["width"]), int(r["seed"])))
assert len(rows) == 6, len(rows)
print(rows[int(sys.argv[1])]["array_index"])
PY
)
python train.py --root "$DM4" --index "$idx" --out "$DM4/g1/run$idx" --no-cooldown --log-every 200
echo G1_RUN_DONE "$idx"
