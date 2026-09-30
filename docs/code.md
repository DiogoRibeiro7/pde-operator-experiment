# Code

```text
elliptic1d/
  config.py      every setting, in one frozen dataclass; quick() for smoke tests
  fields.py      random coefficient families: sample, evaluate a and log a anywhere
  solver.py      exact reference solution; finite-volume solver (cell averages or point values),
                 batched in NumPy and compiled in JAX; linear interpolation between grids
  nets.py        MLP and initialisation
  train.py       one Adam loop for every model, plus L-BFGS for the PINN
  pinn.py        PINN for one field, strong form or flux form, exact boundary conditions
  deeponet.py    branch/trunk DeepONet
  fno.py         1D Fourier neural operator with resolution-independent padding
  operators.py   training and evaluating DeepONet and FNO on (a, u) pairs
run_experiment.py        the main experiment: sweep, shift, resolution, PINNs, timing
check_reference.py       how exact the exact solution is
baselines.py             both solver variants on the common grid; matching grid sizes
ablation_log_input.py    log a against a as the operators' input
timing.py                one timing protocol for every method, with the saved models
make_figures.py          figures and derived numbers
tools/mkdocs_hooks.py    builds the site's result tables from results/*.json
tests/                   solver accuracy and order, model shapes and training, repository consistency
```

## Using the pieces

```python
import numpy as np
from elliptic1d import fields, solver
from elliptic1d.config import TRAIN, SHIFTS

rng = np.random.default_rng(0)
p = fields.sample(TRAIN, rng, 10)          # ten coefficient fields
u_exact = solver.exact(p, 129)             # exact solution on 129 points
u_fd = solver.fd(p, 129)                   # cell-average finite differences
u_pv = solver.fd_nodal(p, 129)             # point-value finite differences
```

Training an operator on your own data:

```python
from elliptic1d import operators
from elliptic1d.config import Config

cfg = Config()
x = solver.grid(cfg.n_grid)
p_train = fields.sample(TRAIN, rng, 1000)
op = operators.fit("fno", cfg, p_train, solver.fd(p_train, cfg.n_grid), x, seed=0)
u_pred = operators.predict(op, cfg, p, cfg.n_grid)
```

A PINN for one field:

```python
from elliptic1d import pinn

params, info = pinn.fit(cfg, fields.take(p, 0), seed=0, form="mixed")
u_pinn = pinn.predict(params, x)
```
