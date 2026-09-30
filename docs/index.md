---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# One equation, four solvers

A finite-difference solver, a physics-informed neural network, a DeepONet and a Fourier neural operator on the same elliptic PDE, scored against an exact solution, on the same grid, with the same clock.

[See the results](results.md){ .md-button .md-button--primary }
[Reproduce it](reproduce.md){ .md-button }

</div>

The problem is \(-(a(x)\,u'(x))' = 1\) on \((0, 1)\) with \(u(0) = u(1) = 0\) and a random coefficient \(a(x) > 0\). In one dimension it has a closed-form solution up to quadrature, so every error reported here is an error against the truth rather than against another model. The classical solver is treated as a competitor, not only as the source of training labels, and it is given no more information about the coefficient than the networks get.

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } **An exact reference**

    ---

    The solution is written in terms of two integrals of \(1/a\), computed exactly for piecewise-constant coefficients and to about \(10^{-13}\) for smooth ones.

    [:octicons-arrow-right-24: The design](design.md)

-   :material-scale-balance:{ .lg .middle } **A fair baseline**

    ---

    Every method is scored on the same 129-point grid and timed from the same input. The solver is also run knowing only the point values of \(a\), exactly what the FNO sees.

    [:octicons-arrow-right-24: The results](results.md)

-   :material-refresh:{ .lg .middle } **Fully reproducible**

    ---

    One command per step, fixed seeds, committed results, and a GitHub Actions workflow that reruns the whole pipeline from scratch.

    [:octicons-arrow-right-24: Reproducing it](reproduce.md)

</div>

## In one paragraph

On the family of coefficients they were trained on, the FNO and the DeepONet reach mean errors that the finite-difference solver matches with 27 and 18 grid points respectively, even when it sees only the point values of the coefficient. Off that family their errors grow by up to fifty-three-fold. A PINN written in the strong form is more accurate than the 129-point solver on training-family fields and fails on rough ones; the same network written in flux form works on every family. Training either operator costs less than training PINNs for three fields, and never pays off against the solver. The tables on the [results page](results.md) are generated from the committed result files at build time.

This repository accompanies Part 3 of a series on scientific machine learning. Part 1: [Beyond PINNs: A Map of Modern Neural Network Paradigms](https://medium.com/@diogo-ribeiro-1975/beyond-pinns-a-map-of-modern-neural-network-paradigms-ee5956e2df51). Part 2: [PINNs vs Neural Operators](https://medium.com/@diogo-ribeiro-1975/pinns-vs-neural-operators-09b3e9806d6e).
