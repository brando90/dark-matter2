#!/bin/bash
# CPU prep. MODE=quick (dev step, 4 CPUs): venv, all shards (sha256-checked), BPE, validation slice in all four
# tokenizers, the first 3 train shards in NeoX (G1) and the first train shard in each own BPE (probe).
# MODE=full (normal partition, 16 CPUs): every train shard in every tokenizer (phase 1's pool). Idempotent.
set -euo pipefail
ENV_SH="${CODE:-}/sherlock/env.sh"; [ -f "$ENV_SH" ] || ENV_SH="$(dirname "$0")/env.sh"   # sbatch runs a spool copy
source "$ENV_SH"
cd "$CODE"
if [ "${MODE:-quick}" = quick ]; then
  python prep.py --root "$DM4" fetch --train-shards "${TRAIN_SHARDS:-8}"
  python prep.py --root "$DM4" bpe
  python prep.py --root "$DM4" tokenize --tokenizers neox --tok-shards 3
  python prep.py --root "$DM4" tokenize --tokenizers bpe8k bpe32k bpe50k --tok-shards 1
else
  python prep.py --root "$DM4" tokenize --tokenizers neox bpe8k bpe32k bpe50k
fi
python prep.py --root "$DM4" stats
echo PREP_OK
