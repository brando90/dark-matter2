#!/bin/bash
#SBATCH -J dm4-probe
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_SKU:L40S
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/probe-%j.log
# Phase 0 throughput probe (<= 0.5 GPU-h): 50 steps per body x vocab (untied: the larger optimizer state), timed over
# steps 20-50; a compiled repeat at two bodies; then Porian's own w128/w160 checkpoints on our validation slice.
set -euo pipefail
source "$CODE/sherlock/env.sh"   # sbatch runs a spool copy: $CODE comes from the submit env
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$DM4/probe/gpu.txt"
for w in 128 160 224 288 320 384 480 576 640 704; do
  for V in neox bpe32k bpe8k; do
    idx=$(python - "$w" "$V" <<'PY'
import csv, sys
w, V = sys.argv[1], {"neox": "50432", "bpe32k": "32768", "bpe8k": "8192"}[sys.argv[2]]
tok = "neox" if V == "50432" else "own-bpe"
for r in csv.DictReader(open("runs.csv")):
    if r["width"] == w and r["vocab"] == V and r["tie"] == "untied" and r["tokenizer"] == tok and r["run"] == "const":
        print(r["array_index"]); break
PY
)
    if [ -z "$idx" ]; then   # no untied row at this body/vocab (phase 2 has tied-8k only): probe that row's shape untied
      idx=$(python -c "import csv;print(next(r['array_index'] for r in csv.DictReader(open('runs.csv')) if r['width']=='$w' and r['run']=='const'))")
      extra="--tie untied --vocab-name $V"
    else extra=""; fi
    python train.py --root "$DM4" --index "$idx" $extra --out "$DM4/probe/w${w}_${V}" --probe-steps 50 --probe-skip 20 \
      --log-every 10 || echo "PROBE_FAIL w$w $V"
  done
done
for w in 128 704; do
  idx=$(python -c "import csv;print(next(r['array_index'] for r in csv.DictReader(open('runs.csv')) if r['width']=='$w' and r['vocab']=='50432' and r['tie']=='untied' and r['tokenizer']=='neox' and r['run']=='const'))")
  python train.py --root "$DM4" --index "$idx" --out "$DM4/probe/w${w}_neox_compiled" --probe-steps 50 --probe-skip 25 \
    --compile --log-every 10 || echo "PROBE_FAIL compiled w$w"
done
python eval_porian.py --root "$DM4" --out "$DM4/probe/porian_check.json" --widths 128 160 || echo "PORIAN_CHECK_FAIL"
python summarize_phase0.py --root "$DM4" probe
echo PROBE_DONE
