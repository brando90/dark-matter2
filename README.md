# dark-matter2

## Project Hypothesis

The Dark Matter Project investigates the “missing terms” in neural scaling 
laws—the higher-order components that standard single-slope power-law models fail to capture. 
Its central aim is to build more accurate predictive laws that can extrapolate 
large-model performance from smaller, cheaper runs with lower error. 
In particular, the project explores extensions of the canonical Chinchilla-style equation

    L(N) = E + A / N^α + B / N^β   with β < α,

which introduces a sub-leading correction term to capture curvature in
scaling-law fits. This formulation arose in discussions with Joshua as a way to 
test whether adding a second power-law term improves back-test and extrapolation 
accuracy over the standard single-term model. However, this is treated as a 
conjecture rather than a commitment—the broader goal is to identify whatever 
functional form most effectively explains residual “dark matter” structure in 
existing scaling ladders (e.g., Chinchilla, Kaplan, Pythia) and 
thereby enable more reliable forecasts of model behavior across compute regimes.


## Brando's Setup in Snap Cluster

Install uv if you don't have it:
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Assuming you are using my `.bashrc` setup, install this repo with `uv`:
```bash
# $HOME points to lfs, due to repos at dfs and symlined from dfs
mkdir -p $HOME/uv_envs 
# if you still have conda
conda deactivate
# create uv env (makes sure all other envs are deactivated)
deactivate
# create uv virtual env
uv venv $HOME/uv_envs/dark-matter2
# activate uv env
source $HOME/uv_envs/dark-matter2/bin/activate
# install the dependencies to the active uv env in lfs for speed
cd $HOME/dark-matter2
# install the dependencies to the active uv env in lfs for speed
uv sync --active
```
