#!/bin/bash
# Build ~/venvs/dm4: /usr/bin/python3.12 venv, the fs_hf lock (torch cu130 from PyTorch's index), then DM-4's extras.
# Idempotent (skips when $VENV/.ready exists). Needs python3.12-venv: not on wheat-01 (submit with --exclude=wheat-01).
set -euo pipefail
source "${DM4_ROOT:-$HOME/dm4}/code/experiments/dm4_embedding_ladder/farmshare/env.sh"
[ -f "$VENV/.ready" ] && { echo "venv ready: $VENV"; exit 0; }
LOCK=$HOME/venvs/fs_hf/requirements.lock.txt
rm -rf "$VENV"
/usr/bin/python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip wheel
grep -v -E '^(-e |#)| @ ' "$LOCK" > "$VENV/fs_hf.lock.txt"
"$VENV/bin/pip" install -q -r "$VENV/fs_hf.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu130 ||
  { echo "lock as is failed: retrying with local version tags stripped"
    sed -E 's/\+cu[0-9]+//' "$VENV/fs_hf.lock.txt" > "$VENV/fs_hf.lock.nolocal.txt"
    "$VENV/bin/pip" install -q -r "$VENV/fs_hf.lock.nolocal.txt" --extra-index-url https://download.pytorch.org/whl/cu130; }
"$VENV/bin/pip" install -q "tokenizers>=0.21" numpy "pyarrow>=17" "pandas>=2.2"
"$VENV/bin/python" -c "import torch, numpy, tokenizers, pyarrow, pandas; print('venv ok torch', torch.__version__, 'tokenizers', tokenizers.__version__, 'numpy', numpy.__version__)"
"$VENV/bin/pip" freeze > "$VENV/requirements.lock.txt"
touch "$VENV/.ready"
