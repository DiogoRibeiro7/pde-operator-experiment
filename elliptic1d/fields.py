"""Random coefficient fields a(x) = exp(g(x)) on [0, 1].

A draw is stored as a small dictionary of numpy arrays, so the same field can
be evaluated anywhere: on a coarse grid, at DeepONet sensors, at PINN
collocation points, or on the fine grid used for the reference solution.

Smooth family
    g(x) = sum_k s_k (xi_k cos(pi k x) + eta_k sin(pi k x)),  xi, eta ~ N(0, 1),
    s_k^2 proportional to exp(-(pi k ell)^2 / 2), normalised so Var g(x) = sigma^2.
    The covariance is sum_k s_k^2 cos(pi k (x - x')): stationary, with a
    Gaussian-shaped spectrum. It is not the squared-exponential covariance,
    because the series has no constant (k = 0) term: for ell = 0.15 the
    correlation is about 0.52 at lag ell and settles near -0.23 beyond lag 0.5.
    So ell is a spectral length scale, not a correlation length. Frequencies are
    multiples of pi, not 2 pi, so realisations are not periodic on [0, 1] and an
    FFT-based model gets no free pass from periodicity.

Piecewise family
    g is constant on the intervals cut by n_pieces - 1 sorted uniform points,
    with i.i.d. N(0, sigma^2) levels.
"""

from __future__ import annotations

import numpy as np

from .config import FieldFamily


def spectral_weights(fam: FieldFamily) -> np.ndarray:
    k = np.arange(1, fam.n_modes + 1)
    s = np.exp(-((np.pi * k * fam.length_scale) ** 2) / 4.0)
    return s * fam.sigma / np.sqrt(np.sum(s**2))


def sample(fam: FieldFamily, rng: np.random.Generator, n: int) -> dict:
    if fam.kind == "smooth":
        return {
            "kind": "smooth",
            "s": spectral_weights(fam),
            "xi": rng.standard_normal((n, fam.n_modes)),
            "eta": rng.standard_normal((n, fam.n_modes)),
        }
    if fam.kind == "piecewise":
        return {
            "kind": "piecewise",
            "edges": np.sort(rng.uniform(0.0, 1.0, (n, fam.n_pieces - 1)), axis=1),
            "levels": fam.sigma * rng.standard_normal((n, fam.n_pieces)),
        }
    raise ValueError(fam.kind)


def size(p: dict) -> int:
    return (p["xi"] if p["kind"] == "smooth" else p["levels"]).shape[0]


def take(p: dict, idx) -> dict:
    """Sub-select draws (idx: int, slice or index array)."""
    idx = np.atleast_1d(np.arange(size(p))[idx])
    out = {"kind": p["kind"]}
    for k, v in p.items():
        if k == "kind":
            continue
        out[k] = v if k == "s" else v[idx]
    return out


def _piece_index(p: dict, x: np.ndarray) -> np.ndarray:
    # number of interior edges at or to the left of x, per draw -> (n, nx)
    return np.sum(x[None, :, None] >= p["edges"][:, None, :], axis=-1)


def log_a(p: dict, x: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """g(x) for every draw, shape (n, len(x)), float64."""
    x = np.asarray(x, dtype=np.float64)
    if p["kind"] == "piecewise":
        return np.take_along_axis(p["levels"], _piece_index(p, x), axis=1)
    w = np.pi * np.arange(1, p["s"].size + 1)
    A, B = p["xi"] * p["s"], p["eta"] * p["s"]
    out = np.empty((A.shape[0], x.size))
    for j in range(0, x.size, chunk):
        ph = np.outer(x[j : j + chunk], w)
        out[:, j : j + chunk] = A @ np.cos(ph).T + B @ np.sin(ph).T
    return out


def a(p: dict, x: np.ndarray) -> np.ndarray:
    return np.exp(log_a(p, x))


def inverse_a_primitives_piecewise(p: dict, x: np.ndarray):
    """Exact I0(x) = int_0^x 1/a and I1(x) = int_0^x s/a(s) ds for piecewise a."""
    n = p["levels"].shape[0]
    e = np.concatenate([np.zeros((n, 1)), p["edges"], np.ones((n, 1))], axis=1)
    b = np.exp(-p["levels"])                                   # 1/a on each piece
    c0 = np.cumsum(b * np.diff(e, axis=1), axis=1)             # I0 at right edges
    c1 = np.cumsum(b * np.diff(e**2, axis=1) / 2.0, axis=1)    # I1 at right edges
    c0 = np.concatenate([np.zeros((n, 1)), c0], axis=1)
    c1 = np.concatenate([np.zeros((n, 1)), c1], axis=1)
    j = _piece_index(p, x)                                     # piece containing x
    left = np.take_along_axis(e, j, axis=1)
    bj = np.take_along_axis(b, j, axis=1)
    I0 = np.take_along_axis(c0, j, axis=1) + bj * (x[None, :] - left)
    I1 = np.take_along_axis(c1, j, axis=1) + bj * (x[None, :] ** 2 - left**2) / 2.0
    return I0, I1
