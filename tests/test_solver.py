"""The reference solution and the finite-difference solver."""

import numpy as np
import pytest
from scipy.integrate import quad

from elliptic1d import fields, solver
from elliptic1d.config import SHIFTS, TRAIN

FAMILIES = (TRAIN,) + SHIFTS
NAMES = [f.name for f in FAMILIES]


def rel_l2(pred, ref):
    return np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(0)
    return {fam.name: fields.sample(fam, rng, 8) for fam in FAMILIES}


def test_smooth_fields_follow_their_series_and_are_normalised(draws):
    p = fields.take(draws["train"], 2)
    x = np.array([0.0, 0.3, 0.71, 1.0])
    k = np.arange(1, p["s"].size + 1)
    direct = [np.sum(p["s"] * (p["xi"][0] * np.cos(np.pi * k * xi) + p["eta"][0] * np.sin(np.pi * k * xi)))
              for xi in x]
    np.testing.assert_allclose(fields.log_a(p, x)[0], direct, rtol=1e-12, atol=1e-12)
    s = fields.spectral_weights(TRAIN)
    assert np.isclose(np.sum(s**2), TRAIN.sigma**2)


def test_piecewise_fields_are_constant_between_their_breakpoints(draws):
    p = fields.take(draws["piecewise"], 0)
    e = np.concatenate([[0.0], p["edges"][0], [1.0]])
    mids = 0.5 * (e[:-1] + e[1:])
    np.testing.assert_allclose(fields.log_a(p, mids)[0], p["levels"][0])


def test_take_and_size(draws):
    p = draws["train"]
    assert fields.size(p) == 8
    assert fields.size(fields.take(p, slice(2, 5))) == 3
    x = solver.grid(33)
    np.testing.assert_allclose(fields.a(fields.take(p, 3), x)[0], fields.a(p, x)[3])


@pytest.mark.parametrize("name", NAMES)
def test_exact_solution_satisfies_boundary_conditions(draws, name):
    u = solver.exact(draws[name], 129, n_fine=4097)
    np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-12)
    assert np.all(u[:, 1:-1] > 0)          # f = 1 > 0 and a > 0: the solution is positive inside


def test_exact_solution_for_constant_coefficient():
    # a = 1 (all levels zero): u = x (1 - x) / 2
    p = {"kind": "piecewise", "edges": np.array([[0.3, 0.7]]), "levels": np.zeros((1, 3))}
    x = solver.grid(65)
    np.testing.assert_allclose(solver.exact(p, 65)[0], x * (1 - x) / 2, atol=1e-15)


@pytest.mark.parametrize("name", NAMES)
def test_cell_average_scheme_is_second_order(draws, name):
    p = draws[name]
    errs = [rel_l2(solver.fd(p, n), solver.exact(p, n, n_fine=8193)).mean() for n in (65, 129, 257)]
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.7), rates


def test_point_value_scheme_is_first_order_on_jumps(draws):
    p = draws["piecewise"]
    errs = [rel_l2(solver.fd_nodal(p, n), solver.exact(p, n)).mean() for n in (129, 513, 2049)]
    rate = np.log2(errs[0] / errs[-1]) / 4          # grids refined by 16
    assert 0.6 < rate < 1.4, rate


def test_single_batch_and_jax_solvers_agree(draws):
    ah = solver.cell_harmonic_a(draws["train"], 129)
    batch = solver.fd_solve_batch(ah)
    for i in range(ah.shape[0]):
        np.testing.assert_allclose(solver.fd_solve_one(ah[i]), batch[i], rtol=1e-12, atol=1e-15)
    u32 = np.asarray(solver.fd_solver_jax()(ah.astype(np.float32), 1.0))
    assert rel_l2(u32, batch).max() < 1e-4


def test_to_grid_is_identity_on_the_same_grid_and_exact_for_lines():
    u = np.random.default_rng(1).standard_normal((3, 33))
    np.testing.assert_allclose(solver.to_grid(u, 33), u)
    x = solver.grid(9)
    np.testing.assert_allclose(solver.to_grid((2 * x + 1)[None], 129)[0], 2 * solver.grid(129) + 1)


@pytest.mark.parametrize("name", ["train", "rough", "high-contrast"])
def test_exact_solution_matches_adaptive_quadrature(draws, name):
    p = fields.take(draws[name], 0)
    inv = lambda s: float(np.exp(-fields.log_a(p, np.array([s]))[0, 0]))  # noqa: E731
    I0 = lambda x: quad(inv, 0, x, limit=200, epsabs=1e-13, epsrel=1e-12)[0]  # noqa: E731
    I1 = lambda x: quad(lambda s: s * inv(s), 0, x, limit=200, epsabs=1e-13, epsrel=1e-12)[0]  # noqa: E731
    C = I1(1.0) / I0(1.0)
    x = solver.grid(9)
    ref = np.array([C * I0(xi) - I1(xi) for xi in x])
    np.testing.assert_allclose(solver.exact(p, 9), ref[None], rtol=1e-10, atol=1e-13)


def test_exact_solution_scales_with_the_right_hand_side(draws):
    for name in ("train", "piecewise"):
        p = draws[name]
        np.testing.assert_allclose(solver.exact(p, 33, n_fine=4097, f=2.5), 2.5 * solver.exact(p, 33, n_fine=4097))
        rel = rel_l2(solver.fd(p, 129, 2.5), solver.exact(p, 129, n_fine=8193, f=2.5)).max()
        assert rel < 1e-3


def test_exact_solution_on_grids_that_do_not_nest(draws):
    """12 does not divide 32768, so the quadrature grid must be refined to contain the 13-point grid."""
    p = fields.take(draws["train"], 1)
    inv = lambda s: float(np.exp(-fields.log_a(p, np.array([s]))[0, 0]))  # noqa: E731
    I0 = lambda x: quad(inv, 0, x, limit=200, epsabs=1e-13, epsrel=1e-12)[0]  # noqa: E731
    I1 = lambda x: quad(lambda s: s * inv(s), 0, x, limit=200, epsabs=1e-13, epsrel=1e-12)[0]  # noqa: E731
    C = I1(1.0) / I0(1.0)
    x = solver.grid(13)
    ref = np.array([C * I0(xi) - I1(xi) for xi in x])
    u = solver.exact(p, 13)
    assert abs(u[0, -1]) < 1e-14
    np.testing.assert_allclose(u[0], ref, rtol=1e-10, atol=1e-13)


def test_nodal_harmonic_mean():
    a = np.array([[1.0, 3.0, 3.0, 0.5]])
    np.testing.assert_allclose(solver.nodal_harmonic_a(a), [[1.5, 3.0, 6.0 / 7.0]])


def test_point_value_scheme_is_second_order_on_smooth_fields(draws):
    p = draws["train"]
    errs = [rel_l2(solver.fd_nodal(p, n), solver.exact(p, n, n_fine=8193)).mean() for n in (65, 129, 257)]
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.8), rates


def test_three_point_grid(draws):
    ah = solver.cell_harmonic_a(draws["train"], 3)
    u = solver.fd_solve_batch(ah)
    np.testing.assert_allclose(u[:, 1], [solver.fd_solve_one(ah[i])[1] for i in range(ah.shape[0])])
    with pytest.raises(ValueError):
        solver.fd_solve_batch(ah[:, :1])
