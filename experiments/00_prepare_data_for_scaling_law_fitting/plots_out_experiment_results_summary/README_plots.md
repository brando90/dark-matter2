# Scaling-Law Plots Guide (experiment_results_summary.csv)

Fitted params: E=0.000, A=5.39, α=0.059, B=3447.10, β=0.417
Fit quality: R²=0.8964, RMSE=0.2995, D/N≈30.4

Checks (no ground-truth targets assumed):
- `residuals.png`: Roughly symmetric around 0; strong skew/funnel implies bias or heteroscedasticity.
- `compute_loss_offset_vs_nooffset.png`: Solid (offset) curve should clearly beat dashed (no-offset); otherwise the irreducible-loss term may be misfit.
- `isoflops_model_size_vs_compute.png`: Log–log line should be straight-ish; slope ~β/(α+β) if the joint law holds. Curvature or extreme slope ⇒ unreliable optimum sizing.
- `isoflops_dataset_size_vs_compute.png`: Complements model frontier; slopes should roughly sum to 1 (C∝N·D). Curvature or mismatch to model plot ⇒ unstable minima.
- `verification.log` / `scaling_laws_fit.log`: Inspect fitted params and R²; use domain intuition to judge plausibility.
- `fit_summary.txt` (if present): One-line recap of fitted params and metrics.

Flag as suspect if residuals funnel/skew, isoFLOPs curves bend or have implausible slopes, or offset vs no-offset curves are indistinguishable.
