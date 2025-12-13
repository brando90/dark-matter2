# Scaling-Law Plots Guide (synthetic)

Fitted params: E=1.922, A=1651.46, α=0.407, B=4638.69, β=0.399
Fit quality: R²=0.9867, RMSE=0.0332, D/N≈19.7

For Chinchilla replication, a correct fit should be near:
E≈1.82, A≈482.0, α≈0.34, B≈2085.0, β≈0.36, D/N≈20.

- `fitted_vs_true_scaling_law.png`: Points should hug y=x on log-log axes; big curvature or spread ⇒ bad fit.
- `chinchilla_verification_residuals.png`: Residuals centered at 0 with symmetric cloud and bell-ish histogram; funnels/skew ⇒ bias.
- `kaplan_chinchilla_compute_loss.png`: Solid (offset) curve should sit on points; its RMSE must be ≪ dashed (no-offset) curve.
- `isoflops_model_size_vs_compute.png`: Log-log line slope ≈ β/(α+β) (~0.5 for Chinchilla); should be straight, not curved.
- `isoflops_dataset_size_vs_compute.png`: Slope ≈ α/(α+β) (~0.5); complements the model-size plot (slopes sum ≈ 1).
- `verification.log` / `scaling_laws_fit.log`: Check printed params vs targets and R²>0.95 for a clean replication.
- `fit_summary.txt` (if present): One-line sanity check of fitted params and metrics.

If D/N is far from ~20 or any slope/plots look off, treat the fit as suspect.
