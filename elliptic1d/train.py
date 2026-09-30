"""One training loop for everything: Adam with cosine decay, run in jitted chunks.

`loss_fn(params, key)` draws whatever it needs (a mini-batch of functions, or
fresh collocation points) from `key`, so the same loop trains a PINN, a
DeepONet and an FNO. Wall-clock time includes JIT compilation.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax


def adam(loss_fn, params, steps: int, lr: float, seed: int, chunk: int = 250):
    chunk = min(chunk, steps)
    assert steps % chunk == 0, "steps must be a multiple of chunk"
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=1e-2))
    state = opt.init(params)

    @jax.jit
    def run(params, state, key):
        def step(carry, _):
            p, s, k = carry
            k, sub = jax.random.split(k)
            loss, g = jax.value_and_grad(loss_fn)(p, sub)
            upd, s = opt.update(g, s, p)
            return (optax.apply_updates(p, upd), s, k), loss

        (params, state, key), losses = jax.lax.scan(step, (params, state, key), None, length=chunk)
        return params, state, key, losses.mean()

    key = jax.random.key(seed)
    history, chunk_times = [], []
    t0 = time.perf_counter()
    for i in range(steps // chunk):
        tc = time.perf_counter()
        params, state, key, loss = run(params, state, key)
        history.append(((i + 1) * chunk, float(loss)))
        chunk_times.append(time.perf_counter() - tc)
    jax.block_until_ready(params)
    total = time.perf_counter() - t0
    # the first chunk includes JIT compilation; estimate it against the others
    compile_est = chunk_times[0] - float(np.median(chunk_times[1:])) if len(chunk_times) > 1 else 0.0
    return params, {"seconds": total, "compile_seconds_est": max(compile_est, 0.0), "history": history}


def lbfgs(loss_fn, params, steps: int, chunk: int = 100):
    """Deterministic full-batch L-BFGS with a zoom line search (optax defaults)."""
    if steps == 0:
        return params, {"seconds": 0.0, "history": []}
    chunk = min(chunk, steps)
    opt = optax.lbfgs()
    state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def run(params, state):
        def step(carry, _):
            p, s = carry
            value, g = value_and_grad(p, state=s)
            upd, s = opt.update(g, s, p, value=value, grad=g, value_fn=loss_fn)
            return (optax.apply_updates(p, upd), s), value

        (params, state), values = jax.lax.scan(step, (params, state), None, length=chunk)
        return params, state, values[-1]

    history = []
    t0 = time.perf_counter()
    for i in range(steps // chunk):
        params, state, value = run(params, state)
        history.append(((i + 1) * chunk, float(value)))
    jax.block_until_ready(params)
    return params, {"seconds": time.perf_counter() - t0, "history": history}


def relative_l2(pred, target):
    """Per-sample ||pred - target|| / ||target|| along the last axis."""
    return jnp.linalg.norm(pred - target, axis=-1) / jnp.linalg.norm(target, axis=-1)
