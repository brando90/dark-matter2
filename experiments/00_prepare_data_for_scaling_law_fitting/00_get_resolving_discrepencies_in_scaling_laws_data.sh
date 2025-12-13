#!/usr/bin/env bash
set -euo pipefail

# Download the resolving-scaling-law-discrepancies run-level data into our local cache.
mkdir -p "$HOME/dark-matter2/data/data_resolving_discrepancies"
cd "$HOME/dark-matter2/data/data_resolving_discrepancies"

# Fetch the experiment_results.pickle.xz artifact (resume downloads if interrupted).
curl -L -C - \
  https://github.com/formll/resolving-scaling-law-discrepancies/raw/main/data/experiment_results.pickle.xz \
  -o experiment_results.pickle.xz

# Summarize the raw dataset schema against the upstream README claims.
python - <<'PY'
import pathlib
import pandas as pd
import textwrap

base = pathlib.Path.home() / "dark-matter2"
data_path = base / "data/data_resolving_discrepancies/experiment_results.pickle.xz"
artifacts = base / "experiments/00_prepare_data_for_scaling_law_fitting/artifacts"
artifacts.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(data_path)

expected = [
    "hparams",
    "warmup",
    "decay",
    "dataset",
    "train/loss",
    "val/loss",
    "ckpt_path",
    "width",
    "depth",
    "val/loss_std",
    "train/batch_time",
    "train/data_time",
    "train/lr",
    "warmup_tokens",
    "beta1",
    "beta2",
    "world_size",
    "lr",
    "seed",
    "seq_len",
    "vocab_size",
    "grad_clip_norm",
    "optimizer",
    "precision",
    "qk_norm",
    "z_loss_coefficient",
    "bs",
    "independent_wd",
    "max_step",
]

actual = list(df.columns)
missing = [c for c in expected if c not in actual]
extra = [c for c in actual if c not in expected]

def sample_type(col: str) -> str:
    return type(df.iloc[0][col]).__name__

type_rows = "\n".join(
    f"- {col}: dtype={df[col].dtype}, sample_value_type={sample_type(col)}"
    for col in actual
)

summary = f"""# Resolving Discrepancies raw data summary

- rows: {len(df)}
- columns: {len(actual)}
- matches upstream README column list: {not missing and not extra}
- missing columns vs README: {missing or 'none'}
- extra columns not in README: {extra or 'none'}

## Column order (actual)
{', '.join(actual)}

## Column types (pandas dtype, sample value type)
{type_rows}

## Key categorical values
- dataset values: {sorted(df['dataset'].unique())}
- hparams values: {sorted(df['hparams'].unique())}
- warmup values: {sorted(df['warmup'].unique())}
- decay values: {sorted(df['decay'].unique())}
"""

(artifacts / "resolving_discrepancies_raw_summary.md").write_text(
    textwrap.dedent(summary)
)
PY

# Inspect the data here and save a summary of the raw data in a .md in `~/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/artifacts`
