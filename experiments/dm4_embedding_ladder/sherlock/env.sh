# DM-4 environment on Sherlock: source it. Own wheel-only venv under $SCRATCH/dm4 (never shared with other projects).
module load python/3.12.1
export DM4=${DM4:-$SCRATCH/dm4}
export VENV=${VENV:-$DM4/venv}
export CODE=${CODE:-$DM4/code/experiments/dm4_embedding_ladder}
if [ ! -f "$VENV/.ready" ]; then
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip &&
  "$VENV/bin/pip" install -q --only-binary=:all: torch==2.6.0 numpy==2.2.6 "tokenizers>=0.21,<0.23" "pyarrow>=17" "pandas>=2.2,<2.4" &&
  "$VENV/bin/python" -c "import torch, numpy, tokenizers, pyarrow, pandas; print('venv ok', torch.__version__)" &&
  touch "$VENV/.ready"
fi
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true
