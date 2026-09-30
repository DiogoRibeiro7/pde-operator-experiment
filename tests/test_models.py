"""The three learners: shapes, boundary conditions, residuals and layers against independent
computations, and that training reduces the error."""

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


def test_fno_padded_period_is_nine_eighths_of_the_domain_on_every_accepted_grid():
    for n in (9, 17, 65, 129, 257, 513, 2049):
        h = 1.0 / (n - 1)
        assert np.isclose((n + fno.padding(n)) * h, 9 / 8)


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


# ----------------------------------------------------------------------------------------------
# Residuals and layers checked against independent computations, not just shapes and trends


def test_strong_residual_equals_minus_the_derivative_of_the_flux():
    """-(a' u' + a u'') - f must equal -(a u')' - f computed by differentiating the flux itself."""
    import jax

    p = fields.take(fields.sample(TRAIN, np.random.default_rng(3), 1), 0)
    fa = pinn.field_arrays(p)
    params = pinn.init(CFG, np.random.default_rng(4), mixed=False)
    du = jax.grad(pinn.u_fn, argnums=1)
    flux = lambda x: pinn.a_and_da(fa, x)[0] * du(params, x)  # noqa: E731
    for x in (0.13, 0.41, 0.77):
        ref = -jax.grad(flux)(jnp.float32(x)) - 1.0
        got = pinn.residual_strong(params, fa, 1.0, jnp.float32(x))
        assert np.isclose(float(got), float(ref), rtol=1e-4, atol=1e-4)


def test_series_derivative_of_the_coefficient():
    p = fields.take(fields.sample(TRAIN, np.random.default_rng(5), 1), 0)
    fa = pinn.field_arrays(p)
    for x in (0.2, 0.5, 0.9):
        h = 1e-4
        fd = (fields.a(p, np.array([x + h]))[0, 0] - fields.a(p, np.array([x - h]))[0, 0]) / (2 * h)
        a, da = pinn.a_and_da(fa, jnp.float32(x))
        assert np.isclose(float(a), fields.a(p, np.array([x]))[0, 0], rtol=1e-5)
        assert np.isclose(float(da), fd, rtol=1e-3, atol=1e-3)


def test_both_residuals_vanish_on_an_exact_solution(monkeypatch):
    """With a = 1 and f = 1 the solution is u = x(1 - x)/2 and the flux q = (1 - 2x)/2.
    A network that outputs exactly that must have zero residual in both forms."""
    def exact_net(params, x):
        return jnp.stack([jnp.asarray(0.5, jnp.float32), (1.0 - 2.0 * x) / 2.0])

    monkeypatch.setattr(pinn, "_net", exact_net)
    const = {"kind": "piecewise", "edges": np.array([[0.3, 0.7]]), "levels": np.zeros((1, 3))}
    fa = pinn.field_arrays(const)
    smooth = {"kind": "smooth", "s": np.ones(2), "xi": np.zeros((1, 2)), "eta": np.zeros((1, 2))}   # a = 1
    for x in (0.1, 0.5, 0.85):
        np.testing.assert_allclose(np.asarray(pinn.residual_mixed(None, fa, 1.0, jnp.float32(x))), 0.0, atol=1e-6)
        assert abs(float(pinn.residual_strong(None, pinn.field_arrays(smooth), 1.0, jnp.float32(x)))) < 1e-6
        # and the residual does see the right-hand side
        assert abs(float(pinn.residual_mixed(None, fa, 2.0, jnp.float32(x))[0]) - 1.0) < 1e-6


def _fno_reference(params, g, x):
    """The same FNO written with explicit DFT sums in NumPy, to check fno.apply."""
    import jax

    g, x = np.asarray(g, np.float64), np.asarray(x, np.float64)
    n = g.shape[1]
    pad, L = fno.padding(n), n + fno.padding(n)
    P = jax.tree_util.tree_map(lambda v: np.asarray(v, np.float64), params)
    h = np.stack([g, np.broadcast_to(x, g.shape)], axis=-1) @ P["lift"]["W"] + P["lift"]["b"]
    h = np.concatenate([h, np.zeros((h.shape[0], pad, h.shape[2]))], axis=1)
    j = np.arange(L)
    for i, layer in enumerate(P["layers"]):
        m = layer["Rr"].shape[0]
        k = np.arange(m)
        c = np.einsum("bjc,kj->bkc", h, np.exp(-2j * np.pi * np.outer(k, j) / L))         # first m DFT terms
        c = np.einsum("bkc,kco->bko", c, layer["Rr"] + 1j * layer["Ri"])
        c[:, 0] = c[:, 0].real                                                           # irfft drops Im c_0
        wts = np.where(k == 0, 1.0, 2.0)[:, None]
        y = np.real(np.einsum("bko,kj->bjo", c, wts * np.exp(2j * np.pi * np.outer(k, j) / L))) / L
        h = y + h @ layer["W"] + layer["b"]
        if i < len(P["layers"]) - 1:
            h = np.asarray(jax.nn.gelu(h))
    h = h[:, :n]
    for layer in P["proj"][:-1]:
        h = np.asarray(jax.nn.gelu(h @ layer["W"] + layer["b"]))
    q = (h @ P["proj"][-1]["W"] + P["proj"][-1]["b"])[..., 0]
    return x * (1.0 - x) * q


def test_fno_matches_an_explicit_dft_implementation():
    cfg = replace(CFG, fno_width=6, fno_layers=2, fno_modes=4)
    params = fno.init(cfg, np.random.default_rng(0))
    params["layers"] = [{**lay, "Ri": lay["Ri"] * 5.0} for lay in params["layers"]]   # make Ri matter
    n = 33
    x = jnp.linspace(0.0, 1.0, n)
    g = jnp.asarray(np.random.default_rng(1).standard_normal((2, n)), jnp.float32)
    got = np.asarray(fno.apply(params, g, x))
    ref = _fno_reference(params, g, x)
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-5 * np.abs(ref).max())


def test_fno_refuses_grids_where_the_padded_period_is_not_nine_eighths():
    params = fno.init(CFG, np.random.default_rng(0))
    for n in (64, 100, 130):
        with pytest.raises(ValueError):
            fno.apply(params, jnp.zeros((1, n), jnp.float32), jnp.linspace(0.0, 1.0, n))


@pytest.mark.parametrize("kind", ["deeponet", "fno"])
def test_operator_training_cuts_the_error_of_the_untrained_model(kind):
    rng = np.random.default_rng(0)
    p = fields.sample(TRAIN, rng, 64)
    x = solver.grid(CFG.n_grid)
    u = solver.fd(p, CFG.n_grid)

    def error(steps):
        cfg = replace(CFG, don_steps=steps, fno_steps=steps)
        op = operators.fit(kind, cfg, p, u, x, seed=0)
        pred = operators.predict(op, cfg, p, cfg.n_grid)
        return float(np.mean(np.linalg.norm(pred - u, axis=1) / np.linalg.norm(u, axis=1)))

    assert error(250) < 0.5 * error(0)


def test_lbfgs_converges_and_compiles_once(caplog):
    import logging

    import jax

    target = jnp.arange(5.0)
    loss = lambda p: jnp.sum((p["w"] - target) ** 2) + 0.1 * jnp.sum(p["w"] ** 4)  # noqa: E731
    jax.config.update("jax_log_compiles", True)
    try:
        with caplog.at_level(logging.WARNING):
            params, info = train.lbfgs(loss, {"w": jnp.zeros(5)}, steps=300, chunk=100)
    finally:
        jax.config.update("jax_log_compiles", False)
    compiles = [r for r in caplog.records if "Compiling" in r.getMessage() and "jit(run)" in r.getMessage()]
    assert len(compiles) == 1, [r.getMessage()[:80] for r in compiles]
    grad = jax.grad(loss)(params)
    assert float(jnp.max(jnp.abs(grad["w"]))) < 1e-4 and len(info["history"]) == 3
