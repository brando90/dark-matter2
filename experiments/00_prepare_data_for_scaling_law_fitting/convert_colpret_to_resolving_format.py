#!/usr/bin/env python3
"""
Convert ColPret (Hitchhiker) scaling data to the resolving_discrepancies CSV schema.

We load the ColPret data via `util.read_data.get_data()` and keep one row per
model (the max tokens_seen with a non-null train_loss). Missing architecture /
training hyperparameters are left empty; we encode num_params via `width` and
set `depth=1`, and encode tokens_seen via `max_step` with `seq_len=bs=1` so
max_step*seq_len*bs reproduces the original tokens_seen.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert ColPret data to resolving_discrepancies column layout")
    parser.add_argument(
        "--colpret-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "ColPret"),
        help="Path to cloned IBM/ColPret repository (default: ../ColPret relative to repo root)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "data_hitchhickers" / "colpret_resolving_format.csv"),
        help="Where to save the converted CSV",
    )
    parser.add_argument(
        "--template-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "data_resolving_discrepancies" / "experiment_results_summary.csv"),
        help="CSV to read column order from (only the header is used)",
    )
    args = parser.parse_args()

    colpret_root = Path(args.colpret_root).expanduser()
    if not colpret_root.exists():
        raise FileNotFoundError(f"ColPret repo not found at {colpret_root}. Clone https://github.com/IBM/ColPret there or pass --colpret-root.")

    sys.path.insert(0, str(colpret_root))
    try:
        from util.read_data import get_data
    except ImportError as e:
        raise ImportError(f"Failed to import util.read_data from {colpret_root}: {e}")

    # ColPret loaders expect to find raw_data/ relative to their repo root
    cwd_before = Path.cwd()
    os.chdir(colpret_root)
    print("📥 Loading ColPret data (this can take ~1-2 minutes)...")
    df = get_data()
    os.chdir(cwd_before)
    print(f"Loaded ColPret dataframe: rows={len(df)}, cols={len(df.columns)}")

    # Use only rows with train_loss and keep the largest tokens_seen per model_name
    df = df[df["train_loss"].notna()].copy()
    idx = df.groupby("model_name")["tokens_seen"].idxmax()
    df = df.loc[idx].reset_index(drop=True)
    print(f"Kept {len(df)} rows (one per model with max tokens_seen and non-null train_loss)")

    template_cols = pd.read_csv(args.template_csv, nrows=0).columns.tolist()
    out_rows = []

    for row in df.itertuples(index=False):
        record = {c: pd.NA for c in template_cols}
        record["dataset"] = getattr(row, "data", "hitchhikers")
        record["hparams"] = getattr(row, "model_type", "hitchhikers")
        record["warmup"] = "unknown"
        record["decay"] = "unknown"
        record["width"] = float(getattr(row, "num_params"))
        record["depth"] = 1.0
        record["warmup_tokens"] = pd.NA
        record["beta1"] = pd.NA
        record["beta2"] = pd.NA
        record["world_size"] = pd.NA
        record["lr"] = pd.NA
        record["seed"] = getattr(row, "seed", pd.NA)
        record["seq_len"] = 1.0
        record["vocab_size"] = pd.NA
        record["grad_clip_norm"] = pd.NA
        record["optimizer"] = "unknown"
        record["precision"] = pd.NA
        record["qk_norm"] = False
        record["z_loss_coefficient"] = pd.NA
        record["bs"] = 1.0
        record["independent_wd"] = pd.NA
        record["max_step"] = float(getattr(row, "tokens_seen"))
        record["ckpt_path"] = getattr(row, "checkpoint", pd.NA)
        loss_val = float(getattr(row, "train_loss"))
        record["final_val_loss"] = loss_val
        record["min_val_loss"] = loss_val
        record["initial_val_loss"] = pd.NA
        record["final_train_loss"] = loss_val
        record["min_train_loss"] = loss_val
        record["total_training_steps"] = pd.NA
        out_rows.append(record)

    out_df = pd.DataFrame(out_rows, columns=template_cols)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved converted ColPret data to {out_path} (rows={len(out_df)}, cols={len(out_df.columns)})")


if __name__ == "__main__":
    main()
