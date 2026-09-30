"""The three learners: shapes, boundary conditions, and that training reduces the loss."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from elliptic1d import deeponet, fields, fno, operators, pinn, solver, train
from elliptic1d.config import SHIFTS, TRAIN, Config

CFG = Config()


def test_deeponet_output_shape_and_boundary_values():
    params = deeponet.init(CFG, np.random.default_rng(0))
    g = jnp.asarray(np.random.default_rng(1).standard_normal((4, CFG.n_sensors)), jnp.float32)
    x = jnp.linspace(0.0, 1.0, 50)
    u = deeponet.apply(params, g, x)
    assert u.shape == (4, 50)
    np.testing.assert_allclose(np.asarray(u[:, [0, -1]]), 0.0, atol=1e-7)


@pytest.mark.parametrize("n", [65, 129, 257])
def test_fno_runs_at_any_resolution_with_exact_boundary_values(n):
    params = fno.init(CFG, np.random.default_rng(0))
    x = jnp.linspace(0.0, 1.0, n)
    g = jnp.asarray(np.random.default_rng(1).standard_normal((3, n)), jnp.float32)
    u = fno.apply(params, g, x)
    assert u.shape == (3, n)
    np.testing.assert_allclose(np.asarray(u[:, [0, -1]]), 0.0, atol=1e-7)


def test_fno_padded_period_is_nine_eighths_of_the_domain_at_every_resolution():
    for n in (65, 129, 257, 513):
        h = 1.0 / (n - 1)
        assert np.isclose((n + fno.padding(n)) * h, 9 / 8)


@pytest.mark.parametrize("kind", ["deeponet", "fno"])
def test_operator_training_reduces_the_loss(kind):
    cfg = replace(CFG, don_steps=250, fno_steps=250)
    rng = np.random.default_rng(0)
    p = fields.sample(TRAIN, rng, 64)
    x = solver.grid(cfg.n_grid)
    op = operators.fit(kind, cfg, p, solver.fd(p, cfg.n_grid), x, seed=0)
    assert op.info["history"][-1][1] < 0.5          # relative L2 loss starts near 1
    pred = operators.predict(op, cfg, fields.take(p, slice(0, 4)), cfg.n_grid)
    assert pred.shape == (4, cfg.n_grid) and np.all(np.isfinite(pred))


@pytest.mark.parametrize("fam, form", [(TRAIN, "strong"), (TRAIN, "mixed"), (SHIFTS[2], "mixed")],
                         ids=["train-strong", "train-flux", "piecewise-flux"])
def test_pinn_training_reduces_the_residual(fam, form):
    cfg = replace(CFG, pinn_adam_steps=500, pinn_lbfgs_steps=0)
    p = fields.take(fields.sample(fam, np.random.default_rng(0), 1), 0)
    params, info = pinn.fit(cfg, p, seed=0, form=form)
    assert info["loss_adam"][-1][1] < info["loss_adam"][0][1]
    u = pinn.predict(params, solver.grid(33))
    assert u[0] == 0.0 and abs(u[-1]) < 1e-7


def test_strong_form_is_refused_for_piecewise_coefficients():
    p = fields.take(fields.sample(SHIFTS[2], np.random.default_rng(0), 1), 0)
    with pytest.raises(ValueError):
        pinn.fit(CFG, p, seed=0, form="strong")


def test_unknown_pinn_form_is_refused():
    p = fields.take(fields.sample(TRAIN, np.random.default_rng(0), 1), 0)
    with pytest.raises(ValueError):
        pinn.fit(CFG, p, seed=0, form="flux")


def test_fno_refuses_grids_too_coarse_for_its_modes():
    params = fno.init(CFG, np.random.default_rng(0))
    x = jnp.linspace(0.0, 1.0, 17)
    with pytest.raises(ValueError):
        fno.apply(params, jnp.zeros((1, 17), jnp.float32), x)


def test_training_steps_must_be_a_multiple_of_the_chunk():
    def loss(p, key=None):
        return jnp.sum(p["w"] ** 2)

    params = {"w": jnp.ones(3)}
    with pytest.raises(ValueError):
        train.adam(loss, params, steps=300, lr=1e-2, seed=0, chunk=250)
    with pytest.raises(ValueError):
        train.lbfgs(loss, params, steps=150, chunk=100)
    p0, info = train.adam(loss, params, steps=0, lr=1e-2, seed=0)
    assert info["history"] == [] and np.allclose(p0["w"], 1.0)
