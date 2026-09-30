"""DeepONet: G(a)(x) ~ sum_k b_k(a) t_k(x) + b_0.

The branch net reads log a(x) at fixed sensor locations; the trunk net reads the
query coordinate x. The output is multiplied by x (1 - x) so the boundary
conditions hold exactly, as for the other two models.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import nets
from .config import Config


def sensors(cfg: Config) -> np.ndarray:
    return np.linspace(0.0, 1.0, cfg.n_sensors)


def init(cfg: Config, rng: np.random.Generator) -> dict:
    hidden = [cfg.don_width] * cfg.don_depth
    return {
        "branch": nets.mlp_init(rng, [cfg.n_sensors] + hidden + [cfg.don_rank]),
        "trunk": nets.mlp_init(rng, [1] + hidden + [cfg.don_rank]),
        "b0": jnp.zeros((), jnp.float32),
    }


def apply(params: dict, a_sensors, x):
    """a_sensors: (B, m) standardised log a at the sensors; x: (n,) -> (B, n)."""
    b = nets.mlp_apply(params["branch"], a_sensors)                       # (B, p)
    t = nets.mlp_apply(params["trunk"], (2.0 * x - 1.0)[:, None], final_act=jnp.tanh)  # (n, p)
    return x * (1.0 - x) * (b @ t.T + params["b0"])
