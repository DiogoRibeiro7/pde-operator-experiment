# Reproduce

## Locally

```bash
git clone https://github.com/DiogoRibeiro7/pde-operator-experiment
cd pde-operator-experiment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Then run the pipeline in order:

| Step | Command | Time on two CPU cores | Writes |
| :--- | :--- | ---: | :--- |
| Smoke test | `python run_experiment.py --quick` | about 2 min | `results-quick/` |
| Main run | `python run_experiment.py` | about 64 min | `results/results.json`, `results/arrays.npz`, `results/models/` |
| Reference check | `python check_reference.py` | seconds | `results/reference_check.json` |
| Solver baselines | `python baselines.py` | under a minute | `results/baselines.json` |
| Input ablation | `python ablation_log_input.py` | about 4 min | `results/ablation_log_input.json` |
| Timing | `python timing.py` | under a minute | `results/timing.json` |
| Figures | `python make_figures.py` | seconds | `figures/`, `results/derived.json` |

The quick run writes to its own folder and never overwrites the committed results. The later steps read the main run's outputs, so run them after it.

`requirements.txt` pins the exact versions used for the committed results (Python 3.11, JAX 0.10.2 on CPU, optax 0.2.8, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9). The accuracy results are deterministic with those versions on the same hardware; timings are not.

## On GitHub Actions

Nothing needs to be run by hand:

- **CI** runs the linter, the tests on Python 3.11 and 3.12, and a quick run of the whole pipeline on every push and pull request, and checks that the committed results were not modified.
- **Reproduce results** (Actions tab, *Run workflow*) runs the complete pipeline from scratch on a GitHub runner, prints the new accuracy numbers next to the committed ones, and uploads `results/` and `figures/` as an artifact.
- **Docs** builds this site with `mkdocs build --strict` and deploys it to GitHub Pages from `main`.
- **Release** (Actions tab, *Run workflow*, with a version number) sets the version, runs the checks, tags, and publishes a GitHub release with the results and figures attached.

## The site

```bash
pip install -e ".[docs]"
mkdocs serve
```

The build copies `figures/` into the site and generates every result table from `results/*.json`, so the site cannot drift from the committed numbers.
