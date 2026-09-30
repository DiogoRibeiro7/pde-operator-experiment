"""Small shared building blocks: parameter initialisation and an MLP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def dense_init(rng: np.random.Generator, n_in: int, n_out: int) -> dict:
    """Glorot-normal weights, zero biases."""
    std = np.sqrt(2.0 / (n_in + n_out))
    return {
        "W": jnp.asarray(std * rng.standard_normal((n_in, n_out)), dtype=jnp.float32),
        "b": jnp.zeros((n_out,), dtype=jnp.float32),
    }


def mlp_init(rng: np.random.Generator, sizes: list[int]) -> list[dict]:
    return [dense_init(rng, i, o) for i, o in zip(sizes[:-1], sizes[1:], strict=True)]


def mlp_apply(layers: list[dict], x, act=jnp.tanh, final_act=None):
    for layer in layers[:-1]:
        x = act(x @ layer["W"] + layer["b"])
    x = x @ layers[-1]["W"] + layers[-1]["b"]
    return x if final_act is None else final_act(x)


def n_params(tree) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(tree)))
