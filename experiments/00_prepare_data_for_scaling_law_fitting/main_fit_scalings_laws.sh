# Run scaling law fit on selected CSV(s)
# Usage:
#   SCALING_DATA_CSV=/path/to/data.csv [SCALING_PLOTS_SUFFIX=name] ./main_fit_scalings_laws.sh
#   ./main_fit_scalings_laws.sh /path/to/data.csv
#   ./main_fit_scalings_laws.sh          # defaults to resolving_discrepancies + Hitchhiker

set -euo pipefail

# Activate uv environment
source "$HOME/uv_envs/dark-matter2/bin/activate"

SCRIPT="$HOME/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/main_fit_scalings_laws.py"
DATA_DIR="${SCALING_DATA_DIR:-$HOME/dark-matter2/data}"

# Build list of CSVs to run
csvs=()
if [ -n "${SCALING_DATA_CSV:-}" ]; then
    csvs+=("$SCALING_DATA_CSV")
elif [ -n "${1:-}" ]; then
    csvs+=("$1")
else
    # Default: primary resolving discrepancies + Hitchhiker/ColPret (if present)
    default_candidates=(
        "$DATA_DIR/data_resolving_discrepancies/experiment_results_summary.csv"
        "$DATA_DIR/data_hitchhikers/colpret_resolving_format.csv"
    )
    for c in "${default_candidates[@]}"; do
        [ -f "$c" ] && csvs+=("$c")
    done
    # Fall back to all top-level CSVs in DATA_DIR if nothing found
    if [ ${#csvs[@]} -eq 0 ]; then
        for c in "$DATA_DIR"/*.csv; do
            [ -f "$c" ] && csvs+=("$c")
        done
    fi
fi

if [ ${#csvs[@]} -eq 0 ]; then
    echo "No CSVs found to process. Set SCALING_DATA_CSV or place CSVs in $DATA_DIR."
    exit 1
fi

echo "Running scaling-law fit for ${#csvs[@]} file(s)..."
for csv in "${csvs[@]}"; do
    suffix="${SCALING_PLOTS_SUFFIX:-$(basename "$csv" .csv)}"
    echo ""
    echo "=========================================="
    echo "Processing: $csv"
    echo "Plots suffix: $suffix"
    echo "=========================================="
    python "$SCRIPT" --data_csv="$csv" --plots_suffix="$suffix"
done

echo ""
echo "=========================================="
echo "DONE - Output folders under experiments/00_prepare_data_for_scaling_law_fitting/plots_out_*"
