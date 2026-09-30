"""Every setting for the experiment, in one place.

The problem is the one-dimensional elliptic equation

    -(a(x) u'(x))' = f(x),   0 < x < 1,   u(0) = u(1) = 0,

with f = 1 and a random coefficient a(x) > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FieldFamily:
    """A distribution over coefficient fields a(x) = exp(g(x)).

    kind="smooth":    g is a stationary Gaussian field on [0, 1] with a
                      squared-exponential spectrum of length scale `length_scale`
                      and pointwise standard deviation `sigma`.
    kind="piecewise": g is constant on `n_pieces` random intervals, with
                      i.i.d. N(0, sigma^2) levels, so a(x) jumps.
    """

    name: str = "train"
    kind: str = "smooth"
    length_scale: float = 0.15
    sigma: float = 0.8
    n_modes: int = 64
    n_pieces: int = 5


TRAIN = FieldFamily()

SHIFTS = (
    FieldFamily(name="rough", length_scale=0.05),
    FieldFamily(name="high-contrast", sigma=1.6),
    FieldFamily(name="piecewise", kind="piecewise"),
)


@dataclass(frozen=True)
class Config:
    # --- problem ------------------------------------------------------------
    f_const: float = 1.0
    n_grid: int = 129                  # training resolution, h = 1/128
    n_fine: int = 32769                # 2^15 + 1 points for the reference solution
    test_resolutions: tuple = (65, 129, 257, 513)
    fd_convergence_grids: tuple = (5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049)

    # --- data ---------------------------------------------------------------
    train_family: FieldFamily = TRAIN
    shift_families: tuple = SHIFTS
    n_train: int = 1000                # headline operator models
    n_train_sweep: tuple = (50, 100, 250, 500, 1000, 2000)
    n_seeds: int = 3                   # independent training runs per setting
    n_test: int = 200
    n_pinn_instances: int = 5          # PINNs trained per family (one per instance)
    seed: int = 20260929

    # --- PINN ----------------------------------------------------------------
    pinn_width: int = 64
    pinn_depth: int = 4                # hidden layers
    pinn_adam_steps: int = 10000
    pinn_lr: float = 3e-3
    pinn_collocation: int = 256        # resampled uniformly at every Adam step
    pinn_lbfgs_steps: int = 1000
    pinn_lbfgs_points: int = 1024      # fixed collocation set for L-BFGS

    # --- DeepONet ------------------------------------------------------------
    n_sensors: int = 65                # every other node of the 129-point grid
    don_width: int = 128
    don_depth: int = 3                 # hidden layers in branch and trunk
    don_rank: int = 64                 # p, the number of basis functions
    don_steps: int = 40000
    don_batch: int = 32
    don_lr: float = 1e-3

    # --- FNO -----------------------------------------------------------------
    fno_width: int = 32
    fno_layers: int = 4
    fno_modes: int = 12
    fno_steps: int = 4000
    fno_batch: int = 32
    fno_lr: float = 2e-3


def quick(cfg: Config) -> Config:
    """A tiny configuration that runs end to end in a minute or two."""
    return replace(
        cfg,
        n_train=100,
        n_train_sweep=(50, 100),
        n_seeds=1,
        n_test=20,
        n_pinn_instances=1,
        pinn_adam_steps=500,
        pinn_lbfgs_steps=50,
        don_steps=500,
        fno_steps=250,
        fd_convergence_grids=(9, 17, 33, 65, 129),
    )


def from_dict(d: dict) -> Config:
    """Rebuild a Config from its JSON form (results.json["config"])."""
    def fam(x: dict) -> FieldFamily:
        return FieldFamily(**x)

    kw = {}
    for k, v in d.items():
        if k == "train_family":
            kw[k] = fam(v)
        elif k == "shift_families":
            kw[k] = tuple(fam(x) for x in v)
        elif isinstance(v, list):
            kw[k] = tuple(v)
        else:
            kw[k] = v
    return Config(**kw)
