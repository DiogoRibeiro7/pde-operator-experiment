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
| Input ablation | `python ablation_log_input.py` | about 3 min | `results/ablation_log_input.json` |
| FNO padding check | `python fno_padding_check.py` | seconds | `results/fno_padding_check.json` |
| Timing | `python timing.py` | about 5 min | `results/timing.json` |
| Figures | `python make_figures.py` | seconds | `figures/`, `results/derived.json` |

The later steps read the main run's outputs, so run them after it. Each of them takes `--results DIR`, so the whole pipeline also runs on a quick run:

```bash
python run_experiment.py --quick                 # refuses to write into results/
python check_reference.py --results results-quick
python baselines.py --results results-quick
python ablation_log_input.py --results results-quick
python fno_padding_check.py --results results-quick
python timing.py --results results-quick
python make_figures.py --results results-quick   # figures go to results-quick/figures/
```

`timing.py` measures the training costs again on the machine it runs on, so that every time it reports comes from one machine; `--reuse-train-times` takes them from `results.json` instead.

`requirements.txt` pins the direct dependencies used for the committed results (Python 3.11, JAX 0.10.2 on CPU, optax 0.2.8, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9), and `requirements-lock.txt` pins every package they pull in. The accuracy results are reproducible bit for bit with those versions on the same CPU model; on other hardware the network-based numbers can differ in the last digits. Timings are machine-dependent.

## On GitHub Actions

Nothing needs to be run by hand:

- **CI** runs on every push to `main` and every pull request: the linter, the tests on Python 3.11 and 3.12, and the whole pipeline on a quick run, and it checks that the committed results and figures were not modified.
- **Reproduce results** (Actions tab, *Run workflow*) runs the complete pipeline from scratch on a GitHub runner, prints the new accuracy numbers next to the committed ones, and uploads the results and figures as an artifact.
- **Docs** builds this site with `mkdocs build --strict` and deploys it to GitHub Pages from `main`.
- **Release** (Actions tab, *Run workflow*, with a version number) sets the version, runs the checks, tags, and publishes a GitHub release with the results and figures attached.

## The site

```bash
pip install -e ".[docs]"
mkdocs serve
```

The build copies `figures/` into the site and generates the result tables, the numbers quoted in the prose and the citation from `results/*.json`, `pyproject.toml` and `CITATION.cff`, so the site follows the committed results without hand-copied numbers.
