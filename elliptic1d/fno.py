"""Fourier neural operator in one dimension (Li et al., 2021), written out in full.

lift (log a, x) -> width channels; L Fourier layers  v <- gelu(K v + W v + b), where
K keeps the lowest `modes` Fourier coefficients and multiplies them by learned
complex matrices; project back to one channel; multiply by x (1 - x).

The problem is not periodic, so the lifted field is zero-padded once, before
the first Fourier layer, and cropped after the last one (the padded region
does not stay zero in between). The padding is a fixed fraction of the grid:
the padded period is exactly 9/8 of the domain whenever n - 1 is a multiple of
8, so a given Fourier mode means the same physical frequency at every such
resolution. That is what makes evaluation at other grids meaningful, and
apply() refuses grids for which it does not hold (every grid used here,
65 to 2049 points, qualifies).

irfft discards the imaginary part of the zero-frequency coefficient, so the
imaginary weights of mode 0 (4 layers x 32 x 32 = 4,096 of the 106,977
parameters) never receive a gradient. The reference implementation behaves
the same way.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import nets
from .config import Config


def init(cfg: Config, rng: np.random.Generator) -> dict:
    c, m = cfg.fno_width, cfg.fno_modes
    scale = 1.0 / (c * c)
    layers = []
    for _ in range(cfg.fno_layers):
        layers.append({
            "Rr": jnp.asarray(scale * rng.random((m, c, c)), jnp.float32),
            "Ri": jnp.asarray(scale * rng.random((m, c, c)), jnp.float32),
            **nets.dense_init(rng, c, c),
        })
    return {
        "lift": nets.dense_init(rng, 2, c),
        "layers": layers,
        "proj": nets.mlp_init(rng, [c, 128, 1]),
    }


def padding(n: int) -> int:
    return max((n - 1) // 8 - 1, 0)


def apply(params: dict, a_grid, x):
    """a_grid: (B, n) standardised log a on a uniform grid; x: (n,) -> (B, n)."""
    B, n = a_grid.shape
    if n < 9 or (n - 1) % 8:
        raise ValueError(f"the FNO needs n - 1 to be a positive multiple of 8 so that the padded period is "
                         f"9/8 of the domain; got n = {n}")
    pad = padding(n)
    n_coef = (n + pad) // 2 + 1
    if params["layers"][0]["Rr"].shape[0] > n_coef:
        raise ValueError(f"a {n}-point grid has only {n_coef} Fourier coefficients, fewer than the "
                         f"{params['layers'][0]['Rr'].shape[0]} modes this FNO keeps")
    h = jnp.stack([a_grid, jnp.broadcast_to(x, a_grid.shape)], axis=-1)
    h = h @ params["lift"]["W"] + params["lift"]["b"]
    h = jnp.pad(h, ((0, 0), (0, pad), (0, 0)))
    L = n + pad
    for i, layer in enumerate(params["layers"]):
        m = layer["Rr"].shape[0]
        hf = jnp.fft.rfft(h, axis=1)
        R = layer["Rr"] + 1j * layer["Ri"]
        low = jnp.einsum("bmi,mio->bmo", hf[:, :m], R)
        hf = jnp.concatenate([low, jnp.zeros((B, hf.shape[1] - m, h.shape[-1]), hf.dtype)], axis=1)
        h = jnp.fft.irfft(hf, n=L, axis=1) + h @ layer["W"] + layer["b"]
        if i < len(params["layers"]) - 1:
            h = jax.nn.gelu(h)
    h = h[:, :n]
    q = nets.mlp_apply(params["proj"], h, act=jax.nn.gelu)[..., 0]
    return x * (1.0 - x) * q
