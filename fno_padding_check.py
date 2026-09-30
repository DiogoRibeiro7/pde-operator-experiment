"""How much does the FNO's first spectral layer depend on the grid, and why?

    python fno_padding_check.py                        # seconds
    python fno_padding_check.py --results results-quick

The FNO is not periodic, so its lifted input is zero-padded before the Fourier
transform. The lifted field does not vanish at x = 0 and x = 1, so the padding
creates jumps, whose discrete Fourier coefficients depend on the grid. This
script applies the first spectral layer of the saved FNO (first run) to the
first training-family test field at several resolutions, and compares each
output with the output on 2,049 points, on a common 65-point grid; then it
repeats the comparison with the lifted input multiplied by x (1 - x), which
vanishes at both ends. Writes fno_padding_check.json into the results folder.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from elliptic1d import fields, fno, runs

ROOT = Path(__file__).resolve().parent
REFERENCE_N = 2049
COMMON_N = 65


def first_spectral_layer(params: dict, scaler: dict, p_one: dict, n: int, vanish: bool) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    g = (fields.log_a(p_one, x) - scaler["g_mean"]) / scaler["g_std"]
    h = jnp.stack([jnp.asarray(g, jnp.float32), jnp.broadcast_to(jnp.asarray(x, jnp.float32), g.shape)], axis=-1)
    h = h @ params["lift"]["W"] + params["lift"]["b"]
    if vanish:
        h = h * jnp.asarray(x * (1.0 - x), jnp.float32)[None, :, None]
    pad = fno.padding(n)
    h = jnp.pad(h, ((0, 0), (0, pad), (0, 0)))
    layer = params["layers"][0]
    m = layer["Rr"].shape[0]
    hf = jnp.fft.rfft(h, axis=1)
    low = jnp.einsum("bmi,mio->bmo", hf[:, :m], layer["Rr"] + 1j * layer["Ri"])
    hf = jnp.concatenate([low, jnp.zeros((1, hf.shape[1] - m, h.shape[-1]), hf.dtype)], axis=1)
    out = np.asarray(jnp.fft.irfft(hf, n=n + pad, axis=1))[0, :n]
    return out[:: (n - 1) // (COMMON_N - 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder written by run_experiment.py")
    args = ap.parse_args()
    res_dir = Path(args.results)
    _, cfg = runs.load(res_dir)
    _, tests = runs.draws(cfg)
    runs.check_tests(tests, res_dir, cfg.n_grid)
    with open(res_dir / "models" / "fno_seed0.pkl", "rb") as fh:
        m = pickle.load(fh)
    params = jax.tree_util.tree_map(jnp.asarray, m["params"])
    p_one = fields.take(tests["train"], 0)
    out = {"note": f"relative L2 difference from the output on {REFERENCE_N} points, on a common "
                   f"{COMMON_N}-point grid; first spectral layer of the first-run FNO; first training test field"}
    for vanish, key in ((False, "input_as_in_the_model"), (True, "input_vanishing_at_the_ends")):
        ref = first_spectral_layer(params, m["scaler"], p_one, REFERENCE_N, vanish)
        out[key] = {}
        for n in (65, 129, 257, 513):
            d = first_spectral_layer(params, m["scaler"], p_one, n, vanish)
            out[key][str(n)] = float(np.linalg.norm(d - ref) / np.linalg.norm(ref))
    with open(res_dir / "fno_padding_check.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
