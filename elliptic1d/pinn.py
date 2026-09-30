"""Physics-informed neural network for one coefficient field.

The network represents one solution u_theta(x) for one fixed a(x). Boundary
conditions are imposed exactly through u = x (1 - x) N_theta(x), so the loss
is the PDE residual alone.

Smooth a: strong form, r = -(a u')' - f = -(a' u' + a u'') - f, with a and a'
evaluated analytically from the field's series.

Piecewise-constant a: the strong form is not defined (a' is a sum of Dirac
masses), so the PINN switches to the first-order flux form with a second
output q ~ a u':  r1 = q' + f,  r2 = u' - q / a. This is the usual repair, and
the fact that one is needed is itself part of the comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import nets, train
from .config import Config


def field_arrays(p: dict) -> dict:
    """Single draw -> JAX arrays (float32) the residual can close over."""
    if p["kind"] == "smooth":
        k = np.arange(1, p["s"].size + 1)
        return {
            "kind": "smooth",
            "w": jnp.asarray(np.pi * k, jnp.float32),
            "A": jnp.asarray(p["xi"][0] * p["s"], jnp.float32),
            "B": jnp.asarray(p["eta"][0] * p["s"], jnp.float32),
        }
    return {
        "kind": "piecewise",
        "edges": jnp.asarray(p["edges"][0], jnp.float32),
        "levels": jnp.asarray(p["levels"][0], jnp.float32),
    }


def a_and_da(fa: dict, x):
    """a(x) and a'(x) at a scalar x for a smooth draw."""
    ph = fa["w"] * x
    g = jnp.sum(fa["A"] * jnp.cos(ph) + fa["B"] * jnp.sin(ph))
    dg = jnp.sum(fa["w"] * (-fa["A"] * jnp.sin(ph) + fa["B"] * jnp.cos(ph)))
    a = jnp.exp(g)
    return a, a * dg


def a_value(fa: dict, x):
    if fa["kind"] == "piecewise":
        return jnp.exp(fa["levels"][jnp.sum(x >= fa["edges"])])
    return a_and_da(fa, x)[0]


def init(cfg: Config, rng: np.random.Generator, mixed: bool):
    sizes = [1] + [cfg.pinn_width] * cfg.pinn_depth + [2 if mixed else 1]
    return nets.mlp_init(rng, sizes)


def _net(params, x):
    return nets.mlp_apply(params, jnp.reshape(2.0 * x - 1.0, (1,)))


def u_fn(params, x):
    return x * (1.0 - x) * _net(params, x)[0]


def q_fn(params, x):
    return _net(params, x)[1]


def residual_strong(params, fa, f, x):
    du = jax.grad(u_fn, argnums=1)
    d2u = jax.grad(du, argnums=1)
    a, da = a_and_da(fa, x)
    return -(da * du(params, x) + a * d2u(params, x)) - f


def residual_mixed(params, fa, f, x):
    r1 = jax.grad(q_fn, argnums=1)(params, x) + f
    r2 = jax.grad(u_fn, argnums=1)(params, x) - q_fn(params, x) / a_value(fa, x)
    return jnp.stack([r1, r2])


def make_loss(fa: dict, f: float, mixed: bool):
    res = residual_mixed if mixed else residual_strong

    def loss_at(params, x):
        r = jax.vmap(lambda xi: res(params, fa, f, xi))(x)
        return jnp.mean(r**2) * (2.0 if mixed else 1.0)

    return loss_at


def fit(cfg: Config, p_one: dict, seed: int, form: str = "auto"):
    """Adam on fresh uniform collocation points each step, then L-BFGS on a fixed set.

    form: "strong", "mixed", or "auto" (strong for smooth a, mixed for piecewise a).
    """
    if form not in ("auto", "strong", "mixed"):
        raise ValueError(f"form must be 'auto', 'strong' or 'mixed', not {form!r}")
    fa = field_arrays(p_one)
    if form == "auto":
        form = "mixed" if fa["kind"] == "piecewise" else "strong"
    if form == "strong" and fa["kind"] == "piecewise":
        raise ValueError("the strong form needs a differentiable coefficient")
    mixed = form == "mixed"
    params = init(cfg, np.random.default_rng(seed), mixed)
    loss_at = make_loss(fa, cfg.f_const, mixed)
    n_col = cfg.pinn_collocation

    def adam_loss(params, key):
        return loss_at(params, jax.random.uniform(key, (n_col,)))

    params, info_a = train.adam(adam_loss, params, cfg.pinn_adam_steps, cfg.pinn_lr, seed)
    m = cfg.pinn_lbfgs_points
    x_fixed = jnp.asarray((np.arange(m) + 0.5) / m, jnp.float32)
    params, info_l = train.lbfgs(lambda p: loss_at(p, x_fixed), params, cfg.pinn_lbfgs_steps)
    return params, {
        "seconds": info_a["seconds"] + info_l["seconds"],
        "seconds_adam": info_a["seconds"],
        "seconds_lbfgs": info_l["seconds"],
        "compile_seconds_est": info_a["compile_seconds_est"],
        "loss_adam": info_a["history"],
        "loss_lbfgs": info_l["history"],
        "n_params": nets.n_params(params),
        "mixed_form": mixed,
    }


def predict(params, x: np.ndarray) -> np.ndarray:
    xs = jnp.asarray(x, jnp.float32)
    return np.asarray(jax.vmap(lambda xi: u_fn(params, xi))(xs), dtype=np.float64)
