"""Locating, loading and regenerating the data of a run of run_experiment.py.

The downstream scripts that regenerate data (check_reference.py, baselines.py,
ablation_log_input.py, fno_padding_check.py, timing.py) go through this module,
so they read the same folder, rebuild the same configuration the run used, and
draw the same fields. make_figures.py only reads the result files.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from . import fields, solver
from .config import Config, from_dict


def load(results_dir: str | Path) -> tuple[dict, Config]:
    """results.json of a run, and the Config that produced it."""
    path = Path(results_dir) / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run run_experiment.py first (or pass --results)")
    r = json.loads(path.read_text())
    return r, from_dict(r["config"])


def pool_size(cfg: Config) -> int:
    return max(cfg.n_train_sweep + (cfg.n_train,))


def draws(cfg: Config) -> tuple[dict, dict]:
    """The training pool and the test fields of every family, in the order run_experiment.py draws them."""
    rng = np.random.default_rng(cfg.seed)
    pool = fields.sample(cfg.train_family, rng, pool_size(cfg))
    fams = {cfg.train_family.name: cfg.train_family, **{f.name: f for f in cfg.shift_families}}
    tests = {name: fields.sample(fam, rng, cfg.n_test) for name, fam in fams.items()}
    return pool, tests


def check_tests(tests: dict, results_dir: str | Path, n_grid: int) -> None:
    """Fail loudly if regenerated test fields differ from the ones a run saved."""
    A = np.load(Path(results_dir) / "arrays.npz")
    x = solver.grid(n_grid)
    for name, p in tests.items():
        if not np.allclose(fields.a(fields.take(p, slice(0, 5)), x), A[f"a_{name}"], rtol=0, atol=1e-12):
            raise RuntimeError(f"regenerated {name} fields do not match {results_dir}/arrays.npz")
        a = fields.a(p, x)
        if not np.allclose(a.max(axis=1) / a.min(axis=1), A[f"contrast_{name}"], rtol=1e-12, atol=0):
            raise RuntimeError(f"regenerated {name} fields do not match {results_dir}/arrays.npz")


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment() -> dict:
    """Software versions and hardware, recorded next to every result that depends on them."""
    import jax
    import optax
    import scipy
    return {"python": sys.version.split()[0], "jax": jax.__version__, "optax": optax.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "cpu": cpu_name(), "cpu_count": os.cpu_count()}
