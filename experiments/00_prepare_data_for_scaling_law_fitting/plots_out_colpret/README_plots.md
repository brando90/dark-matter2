# Scaling-Law Plots Guide (colpret)

Fitted params: E=1.208, A=52.95, α=0.199, B=6630.07, β=0.392
Fit quality: R²=0.9696, RMSE=0.0768, D/N≈71.4

If this is the synthetic Chinchilla replication, expected targets are E≈1.82, A≈482, α≈0.34, B≈2085, β≈0.36, D/N≈20.
For Hitchhiker/ColPret or other datasets, these targets do not apply—use plot shapes and R² to judge fit quality.

- `fitted_vs_true_scaling_law.png`: Points should hug y=x on log-log axes; big curvature or spread ⇒ bad fit.
- `chinchilla_verification_residuals.png`: Residuals centered at 0 with symmetric cloud and bell-ish histogram; funnels/skew ⇒ bias.
- `kaplan_chinchilla_compute_loss.png`: Solid (offset) curve should sit on points; its RMSE must be ≪ dashed (no-offset) curve.
- `isoflops_model_size_vs_compute.png`: Log-log line slope should be plausible (≈β/(α+β) if joint law holds); should be straight, not curved.
- `isoflops_dataset_size_vs_compute.png`: Complementary to model-size plot; slopes should sum to ≈1 if C∝N·D holds.
- `verification.log` / `scaling_laws_fit.log`: Check fitted params and R². For non-Chinchilla data, ignore ✓/✗ against Chinchilla targets.
- `fit_summary.txt` (if present): One-line sanity check of fitted params and metrics.

If D/N is far from ~20 or any slope/plots look off, treat the fit as suspect.
