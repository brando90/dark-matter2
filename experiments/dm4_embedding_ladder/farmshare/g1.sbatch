#!/bin/bash
#SBATCH -J dm4-g1r
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-5%4
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/%x-%A_%a.log
# G1 rerun after PREREG Amendment 2 (windows shuffled within 1,024-step chunks): untied-50k NeoX (Porian's exact config)
# at w128 and w160, seeds 0-2, constant LR to D/N = 20, evaluated at D/N = 2.5, 5, 10, 20. Constant-LR leg only.
# FarmShare L40S, QOS gpu (4 running / 32 submitted per user across all experiments: array %4). Resumable: train.py
# checkpoints every 30 min to $OUT/ckpt.pt and --requeue restarts it from there.
# Submit: mkdir -p ~/dm4/logs && cd ~/dm4 && git -C code pull && sbatch code/experiments/dm4_embedding_ladder/farmshare/g1.sbatch
set -euo pipefail
source "${DM4_ROOT:-$HOME/dm4}/code/experiments/dm4_embedding_ladder/farmshare/env.sh"   # spool copy: fixed path
[ -f "$VENV/.ready" ] || { echo "no venv: run farmshare/prep.sbatch first"; exit 1; }
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
idx=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open("runs.csv")) if r["width"] in ("128", "160") and r["vocab"] == "50432"
        and r["tie"] == "untied" and r["tokenizer"] == "neox" and r["run"] == "const"]
rows.sort(key=lambda r: (int(r["width"]), int(r["seed"])))
assert len(rows) == 6, len(rows)
print(rows[int(sys.argv[1])]["array_index"])
PY
)
python train.py --root "$DM4" --index "$idx" --out "$DM4/g1r/run$idx" --no-cooldown --log-every 200
echo G1_RUN_DONE "$idx"
