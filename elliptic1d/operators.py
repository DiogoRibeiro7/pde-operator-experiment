"""Training and evaluating the two operator learners on (a, u) pairs.

Both learn from the same data: coefficient fields a_i drawn from the training
family and finite-difference solutions u_i on the 129-point grid. Both
minimise the mean relative L2 error.

Both read the log-coefficient g = log a rather than a itself, because a is
log-normal and heavily skewed; ablation_log_input.py measures what that is
worth. g is standardised with one global mean and standard deviation from
the training set, a choice that does not depend on
the grid and therefore works at any resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import deeponet, fields, fno, nets, train
from .config import Config

MODELS = {"deeponet": deeponet, "fno": fno}


@dataclass
class Scaler:
    g_mean: float
    g_std: float
    u_scale: float

    @classmethod
    def fit(cls, g_grid: np.ndarray, u: np.ndarray) -> Scaler:
        return cls(float(g_grid.mean()), float(g_grid.std()), float(np.sqrt(np.mean(u**2))))

    def g(self, arr: np.ndarray):
        return jnp.asarray((arr - self.g_mean) / self.g_std, jnp.float32)


@dataclass
class Operator:
    kind: str
    params: dict
    scaler: Scaler
    info: dict
    feature: str = "log_a"


FEATURES = {"log_a": fields.log_a, "a": fields.a}


def model_input(kind: str, cfg: Config, p: dict, n: int, feature: str = "log_a") -> np.ndarray:
    """What each model reads: log a (or a, for the ablation) at the fixed sensors
    (DeepONet) or on the n-point grid (FNO)."""
    pts = deeponet.sensors(cfg) if kind == "deeponet" else np.linspace(0.0, 1.0, n)
    return FEATURES[feature](p, pts)


def fit(kind: str, cfg: Config, p_train: dict, u: np.ndarray, x: np.ndarray, seed: int,
        feature: str = "log_a") -> Operator:
    mod = MODELS[kind]
    scaler = Scaler.fit(FEATURES[feature](p_train, x), u)
    inputs = scaler.g(model_input(kind, cfg, p_train, x.size, feature))
    targets = jnp.asarray(u / scaler.u_scale, jnp.float32)
    xs = jnp.asarray(x, jnp.float32)
    batch = cfg.don_batch if kind == "deeponet" else cfg.fno_batch
    steps = cfg.don_steps if kind == "deeponet" else cfg.fno_steps
    lr = cfg.don_lr if kind == "deeponet" else cfg.fno_lr
    N = inputs.shape[0]

    def loss(params, key):
        idx = jax.random.randint(key, (batch,), 0, N)
        return jnp.mean(train.relative_l2(mod.apply(params, inputs[idx], xs), targets[idx]))

    params = mod.init(cfg, np.random.default_rng(seed))
    params, info = train.adam(loss, params, steps, lr, seed)
    info["n_params"] = nets.n_params(params)
    return Operator(kind, params, scaler, info, feature)


_apply_jit = {k: jax.jit(m.apply) for k, m in MODELS.items()}


def predict(op: Operator, cfg: Config, p: dict, n: int, chunk: int = 500) -> np.ndarray:
    """Predictions on the uniform n-point grid, shape (draws, n), float64."""
    f = _apply_jit[op.kind]
    xs = jnp.asarray(np.linspace(0.0, 1.0, n), jnp.float32)
    g = model_input(op.kind, cfg, p, n, op.feature)
    out = [np.asarray(f(op.params, op.scaler.g(g[j:j + chunk]), xs))
           for j in range(0, g.shape[0], chunk)]
    return np.concatenate(out, axis=0).astype(np.float64) * op.scaler.u_scale
