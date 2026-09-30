"""The reference solution and the finite-difference solver."""

import numpy as np
import pytest

from elliptic1d import fields, solver
from elliptic1d.config import SHIFTS, TRAIN

FAMILIES = (TRAIN,) + SHIFTS


def rel_l2(pred, ref):
    return np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(0)
    return {fam.name: fields.sample(fam, rng, 8) for fam in FAMILIES}


def test_fields_are_positive_and_normalised(draws):
    x = solver.grid(257)
    for p in draws.values():
        assert np.all(fields.a(p, x) > 0)
    s = fields.spectral_weights(TRAIN)
    assert np.isclose(np.sum(s**2), TRAIN.sigma**2)


def test_take_and_size(draws):
    p = draws["train"]
    assert fields.size(p) == 8
    assert fields.size(fields.take(p, slice(2, 5))) == 3
    x = solver.grid(33)
    np.testing.assert_allclose(fields.a(fields.take(p, 3), x)[0], fields.a(p, x)[3])


@pytest.mark.parametrize("name", [f.name for f in FAMILIES])
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


@pytest.mark.parametrize("name", [f.name for f in FAMILIES])
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
