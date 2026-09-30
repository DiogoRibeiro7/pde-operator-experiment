"""Reference solutions and the classical finite-difference solver.

Exact solution. Integrating -(a u')' = f once gives the flux a u' = C - F(x),
with F(x) = int_0^x f. With f = 1, F(x) = x, so

    u(x) = C I0(x) - I1(x),   I0(x) = int_0^x 1/a,   I1(x) = int_0^x s/a(s) ds,

and u(1) = 0 fixes C = I1(1) / I0(1). The only approximation is the
quadrature for I0 and I1: exact for piecewise-constant a, and cumulative
Simpson on 2^15 + 1 points for smooth a.

Finite differences. A conservative (finite-volume) scheme on a uniform grid,

    -(a_{i+1/2} (u_{i+1} - u_i) - a_{i-1/2} (u_i - u_{i-1})) / h^2 = f,

where a_{i+1/2} is the harmonic mean of a over the cell [x_i, x_{i+1}]. That
choice is the standard one for heterogeneous media: it keeps the scheme
second-order accurate even when a jumps inside a cell.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.linalg import solve_banded

from . import fields

GL_T, GL_W = np.polynomial.legendre.leggauss(4)   # 4-point Gauss-Legendre on [-1, 1]


def grid(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def exact(p: dict, n: int, n_fine: int = 32769, chunk: int = 64) -> np.ndarray:
    """Reference solution on the uniform n-point grid, shape (draws, n), float64."""
    x = grid(n)
    if p["kind"] == "piecewise":
        I0, I1 = fields.inverse_a_primitives_piecewise(p, x)
        I0e, I1e = fields.inverse_a_primitives_piecewise(p, np.array([1.0]))
        return (I1e / I0e) * I0 - I1
    if (n_fine - 1) % (n - 1):
        raise ValueError("n - 1 must divide n_fine - 1 so the grids nest")
    step = (n_fine - 1) // (n - 1)
    xf = grid(n_fine)
    out = np.empty((fields.size(p), n))
    for j in range(0, fields.size(p), chunk):
        b = np.exp(-fields.log_a(fields.take(p, slice(j, j + chunk)), xf))
        I0 = cumulative_simpson(b, x=xf, axis=1, initial=0.0)
        I1 = cumulative_simpson(b * xf, x=xf, axis=1, initial=0.0)
        u = (I1[:, -1:] / I0[:, -1:]) * I0 - I1
        out[j : j + chunk] = u[:, ::step]
    return out


def cell_harmonic_a(p: dict, n: int) -> np.ndarray:
    """a_{i+1/2} = h / int_{x_i}^{x_{i+1}} 1/a, shape (draws, n - 1)."""
    x = grid(n)
    h = 1.0 / (n - 1)
    if p["kind"] == "piecewise":
        I0, _ = fields.inverse_a_primitives_piecewise(p, x)
        return h / np.diff(I0, axis=1)
    mid = 0.5 * (x[:-1] + x[1:])
    pts = (mid[:, None] + 0.5 * h * GL_T[None, :]).ravel()
    inv = np.exp(-fields.log_a(p, pts)).reshape(fields.size(p), n - 1, GL_T.size)
    return 1.0 / (0.5 * inv @ GL_W)


def fd_solve_one(a_half: np.ndarray, f: float = 1.0) -> np.ndarray:
    """One tridiagonal solve with LAPACK (the per-instance cost of a classical solve)."""
    n = a_half.size + 1
    h2 = (1.0 / (n - 1)) ** 2
    ab = np.zeros((3, n - 2))
    ab[0, 1:] = -a_half[1:-1] / h2
    ab[1, :] = (a_half[:-1] + a_half[1:]) / h2
    ab[2, :-1] = -a_half[1:-1] / h2
    u = np.zeros(n)
    u[1:-1] = solve_banded((1, 1), ab, np.full(n - 2, f))
    return u


def fd_solve_batch(a_half: np.ndarray, f: float = 1.0) -> np.ndarray:
    """The same scheme, solved for a whole batch at once with the Thomas algorithm."""
    B, m = a_half.shape
    n = m + 1
    h2 = (1.0 / (n - 1)) ** 2
    lower = -a_half[:, 1:-1] / h2           # sub-diagonal (length n - 3)
    diag = (a_half[:, :-1] + a_half[:, 1:]) / h2
    upper = lower                           # symmetric
    d = np.full((B, n - 2), f, dtype=np.float64)
    c = np.empty((B, n - 3))
    c[:, 0] = upper[:, 0] / diag[:, 0]
    d[:, 0] = d[:, 0] / diag[:, 0]
    for i in range(1, n - 2):
        denom = diag[:, i] - lower[:, i - 1] * c[:, i - 1]
        if i < n - 3:
            c[:, i] = upper[:, i] / denom
        d[:, i] = (d[:, i] - lower[:, i - 1] * d[:, i - 1]) / denom
    u = np.zeros((B, n))
    u[:, n - 2] = d[:, -1]
    for i in range(n - 4, -1, -1):
        d[:, i] = d[:, i] - c[:, i] * d[:, i + 1]
        u[:, i + 1] = d[:, i]
    return u


def fd(p: dict, n: int, f: float = 1.0) -> np.ndarray:
    return fd_solve_batch(cell_harmonic_a(p, n), f)


def nodal_harmonic_a(a_nodes: np.ndarray) -> np.ndarray:
    """a_{i+1/2} from point values only: the harmonic mean of the two neighbouring
    nodal values. This variant sees exactly the information an FNO sees on the
    same grid (a at the nodes), and nothing about a between them."""
    lo, hi = a_nodes[..., :-1], a_nodes[..., 1:]
    return 2.0 * lo * hi / (lo + hi)


def fd_nodal(p: dict, n: int, f: float = 1.0) -> np.ndarray:
    return fd_solve_batch(nodal_harmonic_a(fields.a(p, grid(n))), f)


def to_grid(u: np.ndarray, n_out: int) -> np.ndarray:
    """Piecewise-linear interpolation of nodal solutions (draws, n) onto the uniform n_out grid."""
    n = u.shape[-1]
    t = grid(n_out) * (n - 1)
    i = np.minimum(np.floor(t).astype(int), n - 2)
    w = t - i
    return (1.0 - w) * u[..., i] + w * u[..., i + 1]


def _thomas_jax(a_half, f):
    """The same tridiagonal solve, compiled with JAX, for like-for-like timing
    against the JAX-compiled neural networks (float32, like them)."""
    import jax
    import jax.numpy as jnp

    B, m = a_half.shape
    n = m + 1
    h2 = (1.0 / (n - 1)) ** 2
    off = -a_half[:, 1:-1] / h2
    zero = jnp.zeros((B, 1), a_half.dtype)
    lower = jnp.concatenate([zero, off], axis=1)          # lower[i] multiplies u_{i-1}
    upper = jnp.concatenate([off, zero], axis=1)          # upper[i] multiplies u_{i+1}
    diag = (a_half[:, :-1] + a_half[:, 1:]) / h2
    rhs = jnp.full_like(diag, f)

    def fwd(carry, z):
        c_prev, d_prev = carry
        lo, di, up, r = z
        den = di - lo * c_prev
        c, d = up / den, (r - lo * d_prev) / den
        return (c, d), (c, d)

    init = (jnp.zeros(B, a_half.dtype), jnp.zeros(B, a_half.dtype))
    _, (c, d) = jax.lax.scan(fwd, init, (lower.T, diag.T, upper.T, rhs.T))

    def bwd(u_next, z):
        ci, di = z
        u = di - ci * u_next
        return u, u

    _, u = jax.lax.scan(bwd, jnp.zeros(B, a_half.dtype), (c, d), reverse=True)
    return jnp.pad(u.T, ((0, 0), (1, 1)))


def fd_solver_jax():
    import jax
    return jax.jit(_thomas_jax)
