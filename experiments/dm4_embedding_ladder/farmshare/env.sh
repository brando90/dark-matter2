# DM-4 environment on FarmShare: source it (login shell, Jupyter terminal or a job). Same layout as Sherlock's
# $SCRATCH/dm4: code in $DM4_ROOT/code, data in $DM4_ROOT/data. Own venv ~/venvs/dm4 = the shared ~/venvs/fs_hf lock
# plus DM-4's extras (build_venv.sh); fs_hf itself is never modified.
export DM4_ROOT=${DM4_ROOT:-$HOME/dm4}
export DM4=$DM4_ROOT                                   # prep.py/train.py read $DM4 (--root)
export CODE=${CODE:-$DM4_ROOT/code/experiments/dm4_embedding_ladder}
export VENV=${VENV:-$HOME/venvs/dm4}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
if [ -f "$VENV/.ready" ]; then
  source "$VENV/bin/activate"
  # as in fs_hf: pip's nvcc for anything that JIT-compiles; flashinfer's sampler JIT fails on FarmShare nodes
  export CUDA_HOME=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true
mkdir -p "$DM4_ROOT/logs"
