# One equation, four solvers

[![CI](https://github.com/DiogoRibeiro7/pde-operator-experiment/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/pde-operator-experiment/actions/workflows/ci.yml)
[![Docs](https://github.com/DiogoRibeiro7/pde-operator-experiment/actions/workflows/docs.yml/badge.svg)](https://diogoribeiro7.github.io/pde-operator-experiment/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A reproducible comparison of a classical finite-difference solver, a physics-informed
neural network (PINN), a DeepONet and a Fourier neural operator (FNO) on one problem:

```
-(a(x) u'(x))' = 1   on (0, 1),   u(0) = u(1) = 0,   a(x) > 0 random
```

The problem is one-dimensional on purpose: the exact solution is available in closed
form up to quadrature, so every error reported is an error against the truth, not
against another model. The classical solver is treated as a competitor on equal terms:
same test fields, same grid, same clock, and, in one variant, no more information about
the coefficient than the networks get.

**Documentation and results:** <https://diogoribeiro7.github.io/pde-operator-experiment/>

This repository accompanies Part 3 of a series on scientific machine learning.
Part 1: [Beyond PINNs: A Map of Modern Neural Network Paradigms](https://medium.com/@diogo-ribeiro-1975/beyond-pinns-a-map-of-modern-neural-network-paradigms-ee5956e2df51) ·
Part 2: [PINNs vs Neural Operators](https://medium.com/@diogo-ribeiro-1975/pinns-vs-neural-operators-09b3e9806d6e).

## What it measures

1. **The solver itself.** Convergence of the finite-difference scheme against the exact
   solution, for every coefficient family, in two variants: one given cell averages of
   1/a, and one given only the point values of a that the FNO sees.
2. **In-distribution accuracy.** DeepONet and FNO trained on 1,000 (a, u) pairs; PINNs
   trained on individual fields; everything evaluated on the same test fields.
3. **Training-set size.** Operators trained on 50 to 2,000 pairs, three runs each.
4. **Distribution shift.** Operators trained on one family of coefficients and tested on
   three others: rougher fields, higher contrast, and piecewise-constant fields with jumps.
5. **Resolution transfer.** Operators trained on a 129-point grid and evaluated on grids
   of 65, 257 and 513 points.
6. **Cost.** Training time, time per query, and the total cost of answering many queries,
   with every method timed from the same input (a at the grid points) to the same output.

## Run it

```bash
pip install -r requirements.txt -c requirements-lock.txt
pip install -e .
python run_experiment.py --quick    # smoke test, about two minutes, writes results-quick/
python run_experiment.py            # main run, about an hour on two CPU cores
python check_reference.py           # quadrature check on the exact solution
python baselines.py                 # both solver variants on the common 129-point grid
python ablation_log_input.py        # log a against a as the operators' input (~3 min)
python fno_padding_check.py         # how the FNO's padding depends on the grid
python timing.py                    # every method timed under one protocol (~5 min)
python make_figures.py              # figures/ and results/derived.json
```

The main run writes `results/results.json`, `results/arrays.npz` (the arrays the
figures draw), `results/models/` (the trained operators) and `results/run.log`. The
other scripts read those and write `reference_check.json`, `baselines.json`,
`ablation_log_input.json`, `fno_padding_check.json`, `timing.json` and `derived.json` next
to them. Each takes
`--results DIR`, so the whole pipeline also runs on `results-quick/`; `--quick` never
writes into `results/`. The committed `results/` and `figures/` are the ones the
article quotes.

Accuracy results are reproducible bit for bit with the pinned versions on the same CPU
model; on other hardware the network-based numbers can differ in the last digits.
Timings are machine-dependent: `timing.py` measures inference and training again on
the machine it runs on, so every time in `timing.json` comes from one machine. (The
inference timings that older runs recorded in `results.json` are superseded by
`timing.json`.)

The whole pipeline can also be run on GitHub: **Actions → Reproduce results → Run
workflow** reruns everything from scratch and uploads the results as an artifact.

## Layout

```
elliptic1d/
  config.py      every setting, in one frozen dataclass
  runs.py        loading a run's results and regenerating its fields
  fields.py      random coefficient families (smooth Gaussian, piecewise constant)
  solver.py      exact reference solution; finite-volume solver (NumPy, and JAX for timing)
  nets.py        MLP and initialisation
  train.py       one Adam loop for everything, plus L-BFGS for the PINN
  pinn.py        PINN, strong form and flux (first-order) form
  deeponet.py    branch/trunk DeepONet
  fno.py         1D Fourier neural operator
  operators.py   training and evaluating DeepONet and FNO on (a, u) pairs
run_experiment.py        the main experiment
check_reference.py       how exact is the exact solution
baselines.py             solver variants on the common grid, matching grid sizes
ablation_log_input.py    log a vs a as operator input
fno_padding_check.py     how the FNO's first layer depends on the grid
timing.py                one timing protocol for every method
make_figures.py          figures and derived numbers
tools/mkdocs_hooks.py    builds the documentation's result tables from results/*.json
tests/                   solver accuracy and order, models, repository consistency
```

## Design choices worth knowing

- **All three networks impose the boundary conditions exactly**, by multiplying their
  output by x(1 − x). No method pays a boundary penalty.
- **The operators read log a, not a.** The coefficient is log-normal and heavily skewed;
  reading a instead doubles the DeepONet's test error and multiplies the FNO's by 1.8
  (`ablation_log_input.py`).
- **The PINN has two forms.** The strong form needs a′(x), which does not exist for
  piecewise-constant coefficients. The flux form (a second output q ≈ a u′, residuals
  q′ + f and u′ − q/a) needs only a(x). Both are run wherever both are defined.
- **The FNO zero-pads its lifted input once**, before the first Fourier layer, so that
  the padded period is always 9/8 of the domain. A given Fourier mode then means the
  same physical frequency at every resolution.
- **Everything is scored on one grid.** The learned models and the solver are all
  evaluated on the same 129-point grid; a solver run on a coarser grid is interpolated
  linearly onto it (`baselines.py`).
- **Timing starts from the same input for everyone**: a NumPy array of a at the grid
  points, and ends with a NumPy array of u. Conversions and each method's own
  preprocessing are inside the timed region; generating the synthetic fields is not. For
  the solver and the DeepONet the faster of two reasonable implementations is reported,
  including a DeepONet whose trunk is evaluated once and cached (`timing.py`).
- **Budgets were calibrated, not searched.** A handful of calibration runs per model on
  the training family; nothing was tuned on the shifted families.

## Contributing and releases

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs the linter, the tests and the whole
pipeline on a quick run on every push to `main` and every pull request; releases are made
by the **Release** workflow.

## Citation

If you use the code or the results, please cite the repository; GitHub's *Cite this
repository* button uses [CITATION.cff](CITATION.cff).

## Licence

[Apache License 2.0](LICENSE).

## Author

Diogo Ribeiro — Faculdade de Media Artes e Design, Universidade Técnica do Porto · ORCID
[0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072) · dfr@esmad.ipp.pt
