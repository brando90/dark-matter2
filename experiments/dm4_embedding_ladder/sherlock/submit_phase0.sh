#!/bin/bash
# DM-4 phase 0 from a Sherlock login node, after the code is at $SCRATCH/dm4/code. The dev prep step takes up to 2 h,
# so start it detached and close the shell if needed:
#   cd $SCRATCH/dm4 && setsid nohup bash code/experiments/dm4_embedding_ladder/sherlock/submit_phase0.sh > submit.log 2>&1 &
# Progress: tail $SCRATCH/dm4/submit.log $SCRATCH/dm4/logs/prep.log; job ids land in $SCRATCH/dm4/jobs_phase0.txt.
# 1) quick CPU prep as a foreground dev step (prints PREP_OK); 2) the full tokenization (normal, CPU), the probe job
# and the G1 array (both gpu, L40S); 3) an afterany CPU job that packs $DM4/phase0_results.tgz (after all three).
# No phase 1 run is launched here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
mkdir -p "$DM4/logs" "$DM4/probe" "$DM4/g1"
cd "$DM4"
if ! grep -q PREP_OK "$DM4/logs/prep.log" 2>/dev/null; then
  srun -p dev -J dm4-prep -c 4 --mem=16G -t 02:00:00 --mail-type=END,FAIL \
    --mail-user=brandojazz@gmail.com,brando.science@gmail.com bash "$HERE/prep_job.sh" > "$DM4/logs/prep.log" 2>&1 || true
  tail -3 "$DM4/logs/prep.log"
  grep -q PREP_OK "$DM4/logs/prep.log" || { echo "PREP not finished: re-run this script (it resumes)"; exit 1; }
fi
full=$(sbatch --parsable --chdir="$DM4" -p normal -c 16 --mem=64G -t 08:00:00 -J dm4-prepfull --export=ALL,MODE=full \
  --mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com -o "$DM4/logs/prepfull-%j.log" \
  "$HERE/prep_job.sh")
p=$(sbatch --parsable --chdir="$DM4" "$HERE/probe.sbatch")
g=$(sbatch --parsable --chdir="$DM4" "$HERE/g1.sbatch")
post=$(sbatch --parsable --chdir="$DM4" -p normal -c 1 --mem=4G -t 00:30:00 -J dm4-p0post \
  --dependency=afterany:$p:$g:$full --mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com \
  -o "$DM4/logs/post-%j.log" --wrap "source $HERE/env.sh; cd $CODE; python summarize_phase0.py --root $DM4 all; \
  cd $DM4 && tar czf phase0_results.tgz data/manifest.json data/bpt.json probe g1/*/result.json logs phase0_summary.json && echo POST_OK")
echo "prepfull=$full probe=$p g1=$g post=$post" | tee "$DM4/jobs_phase0.txt"
