#!/bin/bash
# Run Chinchilla scaling law fit on one or all data CSVs, saving outputs to per-family folders

# Activate uv environment
source "$HOME/uv_envs/dark-matter2/bin/activate"

DATA_DIR="$HOME/dark-matter2/data"
SCRIPT="$HOME/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/main_chinchilla_verification.py"

if [ -n "$1" ]; then
    # Single file mode: run on provided CSV
    python "$SCRIPT" --mode=dryrun --data_csv="$1" --plots_suffix="$(basename "$1" .csv)"
else
    # Batch mode: run on all CSVs in data dir
    echo "Running on all data files in $DATA_DIR"
    for csv in "$DATA_DIR"/*.csv; do
        family=$(basename "$csv" .csv)
        echo ""
        echo "=========================================="
        echo "Processing: $family"
        echo "=========================================="
        python "$SCRIPT" --mode=dryrun --data_csv="$csv" --plots_suffix="$family"
    done
    echo ""
    echo "=========================================="
    echo "DONE - All output folders:"
    ls -d "$HOME/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/plots/plots_out_"* 2>/dev/null || echo "(none yet)"
fi
