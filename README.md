# Dark Matter 2: the missing terms of scaling laws

Scaling laws let us predict how a large model will do from a ladder of smaller, cheaper runs. The standard laws use
one power law per resource (parameters N, tokens D) and fit a single slope. This project asks what that picture
misses: the "dark matter" of scaling laws, meaning terms we feel in the small-model end of a ladder but don't
model. If the missing terms are real and measurable, small runs become trustworthy for extrapolation, and tiny models
go back on the curve instead of being dropped as outliers.

The project grew out of our Schmidt Sciences SAFE-AI 2024 proposal, **"Beyond Simple Scaling: A Multi-Dimensional
Family of Scaling Laws for Predicting Model Capabilities Under Pre-Training Interventions"**:
[grant proposal (Google Doc)](https://docs.google.com/document/d/1nHuDJUzaKybPVyJscDlqaGOM9QVHryvy5LI-HvrzCIo/edit).
The Doc is not public. Ask a team member below for access.

## Motivation (from the proposal)

For the past half decade, scaling laws have predicted how frontier model capabilities change with scale. But
pretraining is no longer plain next-token prediction on a fixed corpus. Modern runs shift data mixtures, inject
synthetic data, apply targeted "tailpatches" at chosen stages, change or mask the loss (e.g. UL2-style multiple
objectives), grow the model, and ramp the batch size during training. Today's laws only see N and D, so they cannot
say what these interventions do to capability or safety, alone or combined. In practice, engineers layer them on by
intuition and trial and error. The proposal's aim is to make interventions **predictable**, with each one treated
as a dimension of the scaling law. That would move safety from reactive (patch after a problem shows up) to
proactive (predict the effect before the big run).

## Hypothesis: a multi-dimensional family of scaling laws

**1. More than one power law in N (the "dark matter" term).** The working conjecture is that loss is a sum of power
laws, not a single one:

```
L(N) = E + A · N^(-α) + B · N^(-β)        with β > α
```

- `E` is the irreducible loss. `A · N^(-α)` is the familiar slow term that dominates at scale.
- `B · N^(-β)` decays fast, so it matters only for small models. It bends the small end of a ladder and is absorbed
  into a distorted single-term fit at large N.
- If it exists, a single-term fit on small models overestimates the slope and mis-extrapolates. A two-term fit (or a
  principled cut of the smallest sizes) should predict large models better.
- Same idea in compute: `L(C) = E + Σ_i a_i · C^(-b_i)`. A variant with a cross term `F · (N·D)^(-p)` is also on
  the list.
- Convention: in this README the dark-matter term is the one with the **larger** exponent (β > α). The current
  `fit_two_term_N_only` in `py_src/scaling_laws_fit.py` constrains the opposite (β = α·sigmoid(v) < α). Aligning
  the code with this convention is an open TODO.

This is a conjecture, not a commitment. The broader goal is to find whichever functional form best explains the
residual structure in real ladders (Chinchilla, Kaplan, Pythia, and others).

**2. Pre-training interventions as extra dimensions.** The proposal's taxonomy of interventions, each a candidate
axis of the law:

| Category | Examples |
| --- | --- |
| Data changes | replacing stale sources with newer ones, re-weighting text / code / domain data, adding curated preference or synthetic data |
| Targeted and timed tailpatching | injecting curated, filtered or synthetic data at specific stages to fix a weakness |
| Modified and masked losses | multiple simultaneous objectives (e.g. UL2), token masking, control tokens for preferred behavior |
| Hyperparameter changes | batch-size scaling, growing model parameters during training |

The cheapest version of this, which we can test on existing data, is whether the fast term changes with the dataset
or the learning-rate schedule (see H4 below).

## Research questions

From the proposal:

1. How does each pre-training intervention affect model performance and safety?
2. What cumulative effects arise when interventions are combined, and how do they interact with standard scaling?
3. Can the effects of interventions be predicted and ranked within one scaling framework, to guide safer training?

The concrete hypotheses we test first, in order:

- **H1, a second term exists:** on real ladders, the two-term law beats the one-term law (Wald test on B > 2 standard
  errors and lower BIC).
- **H2, small-to-large predictability:** a one-term fit on small models mis-extrapolates; the two-term fit predicts
  the large end better (hold-out and rolling-origin tests).
- **H3, reverse ladder:** fit the large end, predict the small end. Systematic positive residuals at small N are the
  signature of the missing term, even when the two-term fit is not significant.
- **H4, the fast term depends on data and schedule:** e.g. RefinedWeb vs OpenWebText2, decay vs constant learning
  rate in the Porian et al. ladder.
- **H5, parameter-count convention:** embedding vs non-embedding counts (and known errors in reported counts) can
  create or erase a small-N term. This confound must be cleared before claiming H1.
- **H6, RL extension:** RL reward or pass@1 against policy size or RL compute is also a sum of power laws (a fast
  term for early format and easy-skill learning, a slow term for real capability), so small RL runs mis-predict large
  ones in the same way.

## Planned experiments

Status key: **planned** means not in this repo yet; **done** links to the results.

| ID | What | Compute | Status |
| --- | --- | --- | --- |
| DM-1 | **Power analysis on synthetic ladders.** Plant `L = E + A N^-α + B N^-β` with known values, add realistic noise, and vary the ladder span and points per decade. Measure when the Wald and BIC tests recover B and when they report a false one. Says what a real ladder must look like before any fit is trusted. | CPU, a few hours | done: [experiments/dm1_power_analysis](experiments/dm1_power_analysis) (RESULTS.md); LRT variant in RESULTS_DM1b.md |
| DM-2 | **Two-term fits on public ladders.** Porian et al. (OpenWebText2 / RefinedWeb), corrected Chinchilla, the CS336 IsoFLOP ladder, the ColPret families (e.g. Chinchilla, LLaMA-style, Pythia). Per ladder: one-term vs two-term fit, SE(B) from the Hessian and from the bootstrap, Hessian condition number, BIC, and the reverse-ladder test. Output: one table answering "does any public ladder show a significant second term?" | CPU, a few hours | done: [experiments/dm2_real_ladders](experiments/dm2_real_ladders) (RESULTS.md) |
| DM-3 | **Tiny-model CPU ladder.** 10-12 sizes × 3 seeds of small transformers on one fixed corpus, dense at the small end where the term should live: the first ladder we own end to end. Sized by DM-1's answer; may be unnecessary if the Porian seeds give enough noise estimates. | ~100-200 CPU-core-hours | planned |
| RL | **RL extension (H6).** Fit the same one-term vs two-term laws to RL reward / pass@1 curves at several policy sizes, with RL compute in place of N. DM-1's power analysis carries over. | depends on the curves | planned |

Grant-scale validation (from the proposal): train ladders at 412M, 1.4B and 6.9B parameters under different
intervention settings, run ablations of single and combined interventions across sizes, and benchmark across NLP,
code generation and mathematical reasoning (pass@k for math and code, cross-entropy for multiple choice, plus
robustness, consistency and bias metrics for safety). The proposal also commits to releasing the fitting framework
as an open-source package.

## Repository layout

```
dark-matter2/
├── py_src/
│   ├── scaling_laws_fit.py           # fitters: joint L(N,D) = E + A/N^α + B/D^β (Huber loss on log residuals,
│   │                                 #   L-BFGS-B), two-term N-only law, likelihood-ratio test, synthetic data;
│   │                                 #   `python py_src/scaling_laws_fit.py` runs a synthetic self-test
│   └── chinchilla_verification.py    # fit + diagnostic plots (residuals, IsoFLOP frontiers, compute-optimal split)
├── experiments/
│   └── 00_prepare_data_for_scaling_law_fitting/
│       ├── 00_get_resolving_discrepencies_in_scaling_laws_data.py  # download + export the Porian et al. data
│       ├── main_fit_scalings_laws.py / .sh   # fit the joint law to a CSV and write plots to plots_out_<suffix>/
│       ├── plots_out_experiment_results_summary/   # plots from an early fit (see Known issues)
│       └── research_papers/          # reference PDFs (Pythia, Cerebras-GPT, Kaplan vs Chinchilla, CS336 A3)
├── data/
│   ├── data_resolving_discrepancies/ # Porian et al. 2024: experiment_results.pickle.xz (977 runs, 16 sizes,
│   │                                 #   ~5M-901M params, RefinedWeb + OpenWebText2) + EXPERIMENTS_RESULTS_SUMMARY.md
│   └── data_hitchhikers_guide/       # ColPret / "Hitchhiker's Guide to Scaling Law Estimation" (IBM): 65 per-family
│                                     #   CSVs + hitchhickers_guide_unified.csv.gz
├── pyproject.toml, uv.lock
└── LICENSE                           # Apache-2.0
```

## How to run what exists

The scripts use hardcoded `~/dark-matter2/...` paths, so clone into your home directory:

```bash
git clone https://github.com/brando90/dark-matter2.git ~/dark-matter2
cd ~/dark-matter2
```

Environment. The scripts here only need numpy, scipy, pandas, matplotlib and requests:

```bash
python3 -m venv ~/venvs/dm2 && source ~/venvs/dm2/bin/activate
pip install numpy scipy pandas matplotlib requests
```

The full `uv` environment also works, but `pyproject.toml` pulls in a large shared stack (torch, vllm and more):

```bash
uv venv ~/uv_envs/dark-matter2 && source ~/uv_envs/dark-matter2/bin/activate
uv sync --active
```

1. **Synthetic self-test of the fitters** (joint law, two-term N-only law, likelihood-ratio test):

   ```bash
   python py_src/scaling_laws_fit.py
   ```

2. **Export the Porian et al. data** to `data/data_resolving_discrepancies/` (`experiment_results_summary.csv`,
   `experiment_results.json`, both gitignored):

   ```bash
   python experiments/00_prepare_data_for_scaling_law_fitting/00_get_resolving_discrepencies_in_scaling_laws_data.py
   ```

3. **Fit the joint law to a CSV** and write plots plus `README_plots.md` to
   `experiments/00_prepare_data_for_scaling_law_fitting/plots_out_<suffix>/`:

   ```bash
   python experiments/00_prepare_data_for_scaling_law_fitting/main_fit_scalings_laws.py \
     --data_csv data/data_resolving_discrepancies/experiment_results_summary.csv --plots_suffix porian
   ```

   `main_fit_scalings_laws.sh` loops the same fit over CSVs (it activates `~/uv_envs/dark-matter2`; run step 2
   first, since its default inputs are generated files).

## Known issues (to fix before DM-2)

- `chinchilla_verification` derives `N_params = width × depth`, which is not a parameter count (it should be about
  `12 · depth · width²` plus embeddings, or the size recorded per run), and it fits all checkpoints and
  hyperparameter settings together. The plots in `plots_out_experiment_results_summary/` come from that fit, so they
  are not a result about the dark-matter term.
- `fit_two_term_N_only` forces β < α (see the convention note above), uses a single starting point, and reports no
  Wald / Hessian standard errors, BIC, bootstrap or hold-out error.
- `pyproject.toml` packages `py_src/dark_matter2`, which doesn't exist, and lists many dependencies this code
  doesn't use.
- `experiments/00_prepare_data_for_scaling_law_fitting/README.md` refers to `main_chinchilla_verification.py`, which
  was removed. Use `main_fit_scalings_laws.py`.

## Team

From the proposal: **Sanmi Koyejo** (PI), **Rylan Schaeffer**, **Brando Miranda** and **Angelina Wang**.
Code in this repo: Brando Miranda, Joshua Kazdan and Elyas Obbad.

## References

- Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models" (Chinchilla).
- Kaplan et al. (2020), "Scaling Laws for Neural Language Models".
- Besiroglu et al. (2024), "Chinchilla Scaling: A replication attempt".
- Porian et al. (2024), "Resolving Discrepancies in Compute-Optimal Scaling of Language Models"
  ([code and data](https://github.com/formll/resolving-scaling-law-discrepancies)).
- Choshen, Zhang and Andreas (2024), "A Hitchhiker's Guide to Scaling Law Estimation" (ColPret data).
- Pearce and Song (2024), "Reconciling Kaplan and Chinchilla Scaling Laws".
- Biderman et al. (2023), "Pythia".
- Dey et al. (2023), "Cerebras-GPT".
