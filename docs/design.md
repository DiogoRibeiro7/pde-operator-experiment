# Design

## The equation and its exact solution

\[
-\big(a(x)\,u'(x)\big)' = f(x), \quad 0 < x < 1, \qquad u(0) = u(1) = 0,
\]

with \(f \equiv 1\) and a random coefficient \(a(x) > 0\). Integrating once gives the flux, \(a u' = C - x\). Dividing by \(a\) and integrating again with \(u(0) = 0\),

\[
u(x) = C\,I_0(x) - I_1(x), \qquad I_0(x) = \int_0^x \frac{ds}{a(s)}, \qquad I_1(x) = \int_0^x \frac{s\,ds}{a(s)},
\]

and \(u(1) = 0\) fixes \(C = I_1(1)/I_0(1)\). For piecewise-constant coefficients the integrals are elementary and computed exactly. For smooth coefficients they are computed with cumulative Simpson's rule on \(2^{15} + 1\) points; halving that grid moves the reference by less than \(2 \times 10^{-13}\) (`check_reference.py`).

## Coefficient families

The coefficient is \(a = e^{g}\). In the smooth families, \(g\) is a stationary Gaussian field written as a cosine–sine series with frequencies \(\pi k\) (so it is not periodic on \([0, 1]\)), a Gaussian-shaped spectrum with spectral length scale \(\ell\), and pointwise standard deviation \(\sigma\). The series has no constant term, so the covariance is not the squared-exponential one and \(\ell\) is not a correlation length: for \(\ell = 0.15\) the correlation is about 0.52 at lag \(\ell\) and settles near \(-0.23\) beyond lag 0.5.

| Family | Used for | Definition | What changes |
| :--- | :--- | :--- | :--- |
| Training | training and testing | smooth, \(\ell = 0.15\), \(\sigma = 0.8\) | — |
| Rough | testing only | smooth, \(\ell = 0.05\), \(\sigma = 0.8\) | oscillates three times as fast, same pointwise values |
| High contrast | testing only | smooth, \(\ell = 0.15\), \(\sigma = 1.6\) | much larger ratio \(\max a / \min a\) within a field |
| Piecewise constant | testing only | five pieces, levels \(\sim N(0, 0.8^2)\) | jumps where the training fields had none |

## Methods

**Finite differences.** A finite-volume scheme on a uniform grid with a tridiagonal system, in two variants that differ only in what they know about \(a\):

- *cell averages*: the harmonic mean of \(a\) over each cell, the standard choice for heterogeneous media; it needs \(a\) between the grid points;
- *point values*: the harmonic mean of \(a\) at the two neighbouring nodes; like the FNO, it is given only point values of \(a\), at its own grid points and nothing in between.

The cell-average scheme is second order on every family. The point-value scheme is second order on the smooth families and first order on piecewise-constant ones. The cell-average variant also produces the operators' training labels.

**PINN.** A tanh network with four hidden layers of 64 units for one field at a time, with the boundary conditions imposed exactly through \(u = x(1-x)\,N_\theta(x)\). It is trained with 10,000 Adam steps on 256 fresh collocation points per step, then 1,000 L-BFGS steps on 1,024 fixed points, in two residual forms:

- *strong form*: \(r = -(a' u' + a u'') - f\), which needs \(a'\) and cannot be written for piecewise-constant \(a\);
- *flux form*: a second output \(q \approx a u'\) and residuals \(q' + f\) and \(u' - q/a\), with no derivative of \(a\).

**DeepONet.** A branch network reading \(\log a\) at 65 fixed sensors and a trunk network reading \(x\), each with three hidden layers of 128 units, combined through a rank-64 inner product.

**FNO.** A lift of \((\log a, x)\) to 32 channels, four Fourier layers keeping the lowest 12 modes, and a projection. The input is zero-padded once so that the padded period is always \(9/8\) of the domain, which makes a given mode mean the same frequency at every resolution.

Both operators learn from 1,000 pairs \((a, u)\) with a relative \(L^2\) loss (40,000 steps for the DeepONet, 4,000 for the FNO), and every operator number is averaged over three independent runs. All networks multiply their output by \(x(1-x)\), so no method pays a boundary penalty.

## Rules of the comparison

1. **Same test fields.** 200 fields per family, drawn once from a fixed seed.
2. **Same grid.** Every method is scored on the 129-point grid; a solver run on a coarser grid is carried onto it by linear interpolation.
3. **Same kind of information.** The point-value solver is given only point values of \(a\), like the FNO. On most grid sizes those points are not the FNO's own, so the results page also reports the stricter comparison restricted to sub-grids of the FNO's grid.
4. **Same clock.** Every method is timed from the same NumPy array of \(a\) at the grid points to a NumPy array of \(u\) at the grid points, conversions and preprocessing included, with training times measured on the same machine; for the solver and the DeepONet the faster of two reasonable implementations is reported.
5. **No tuning on the test families.** Training budgets were set with a few calibration runs on the training family only; there was no hyperparameter search for any method.
