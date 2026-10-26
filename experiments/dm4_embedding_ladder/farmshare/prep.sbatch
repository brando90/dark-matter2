#!/bin/bash
#SBATCH -J dm4-prep-fs
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/%x-%j.log
# DM-4 CPU data prep on FarmShare, the same layout as Sherlock's $SCRATCH/dm4/data: venv, RefinedWeb shards 0-11
# (sha256-checked), val = shard 0, BPE 50,432 (+ 8k/32k prefixes) from shards 1-3, uint16 memmaps of val + train
# shards 4-11 in neox/bpe8k/bpe32k/bpe50k, data/bpt.json, then the comparison with the repo's data_phase0/bpt.json
# (exit 3 = bytes/token differ at 3 decimals -> FAIL mail). Idempotent: a killed job just re-runs.
# Submit: mkdir -p ~/dm4/logs && cd ~/dm4 && sbatch code/experiments/dm4_embedding_ladder/farmshare/prep.sbatch
set -euo pipefail
FS=${DM4_ROOT:-$HOME/dm4}/code/experiments/dm4_embedding_ladder/farmshare   # sbatch runs a spool copy
bash "$FS/build_venv.sh"
source "$FS/env.sh"
cd "$CODE"
python prep.py --root "$DM4" fetch --train-shards 8
python prep.py --root "$DM4" bpe
python prep.py --root "$DM4" tokenize --tokenizers neox bpe8k bpe32k bpe50k
python prep.py --root "$DM4" stats > /dev/null
du -sh "$DM4/data" "$VENV"
python farmshare/compare_bpt.py "$DM4/data/bpt.json" data_phase0/bpt.json --out "$DM4/data/bpt_compare.json"
echo PREP_OK
