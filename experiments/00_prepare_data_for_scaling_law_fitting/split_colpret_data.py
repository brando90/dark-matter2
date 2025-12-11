"""
Download the ColPret dataset and save one CSV per model family/discrete option.

Families are determined by `scaled_set` when available, otherwise `model_type`.
Outputs are written to the repo's `data/` directory, replacing any existing files there.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


COLPRET_REPO_URL = "https://github.com/IBM/ColPret"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data"


def slugify(name: str) -> str:
    """Make a filesystem-friendly name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("_") or "unnamed"


def load_colpret_df(repo_url: str = COLPRET_REPO_URL) -> pd.DataFrame:
    """Clone ColPret into a temp dir and return the combined dataframe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "colpret"
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_path)], check=True)

        read_data_path = repo_path / "util" / "read_data.py"
        sys.path.insert(0, str(repo_path))
        spec = importlib.util.spec_from_file_location("colpret_read_data", read_data_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        original_cwd = Path.cwd()
        try:
            os.chdir(repo_path)
            df = module.get_data()
        finally:
            os.chdir(original_cwd)
    return df


def family_key(row: pd.Series) -> str:
    """Choose the grouping key, preferring scaled_set over model_type."""
    if pd.notna(row.get("scaled_set")) and str(row.get("scaled_set")).strip():
        return str(row["scaled_set"])
    if pd.notna(row.get("model_type")) and str(row.get("model_type")).strip():
        return str(row["model_type"])
    return "unknown_family"


def clear_output_dir(path: Path) -> None:
    """Remove existing files/dirs in the output directory."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def write_family_splits(df: pd.DataFrame, out_dir: Path) -> Iterable[Path]:
    """Write one CSV per family and return the written paths."""
    df = df.copy()
    df["family"] = df.apply(family_key, axis=1)

    written = []
    for family_name, family_df in df.groupby("family"):
        filename = f"colpret_{slugify(str(family_name))}.csv"
        dest = out_dir / filename
        family_df.to_csv(dest, index=False)
        written.append(dest)
    return written


def main() -> None:
    clear_output_dir(OUTPUT_DIR)
    df = load_colpret_df()
    written = write_family_splits(df, OUTPUT_DIR)
    print(f"Wrote {len(written)} files to {OUTPUT_DIR}")
    for path in sorted(written):
        print(f"- {path.name}")


if __name__ == "__main__":
    sys.exit(main())
