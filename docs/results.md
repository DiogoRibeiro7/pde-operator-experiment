# Results

Every table on this page is generated from the files in `results/` when the site is built, so it always agrees with the committed results. Errors are relative \(L^2\) errors against the exact solution on the 129-point grid.

## Mean error by method and coefficient family

<!-- results:accuracy -->

The operators were trained only on the training family. PINNs are trained separately for each of five fields per family; the strong form cannot be written for piecewise-constant coefficients.

![Instances and pointwise errors](figures/fig1_instances_and_errors.png)
*One field from each family (top) and the absolute error of each method on it (bottom).*

## Accuracy measured in grid points

How many grid points the solver needs to match each operator's mean error on the training family, on the common grid. The last column shows the answer under the easier but less fair convention of scoring the solver at its own nodes.

<!-- results:matching -->

![Accuracy ladder](figures/fig2_accuracy_ladder.png)

## Training-set size

Mean error on the training family, averaged over three runs at each size.

<!-- results:sweep -->

![Training-set size](figures/fig3_training_size.png)

## Distribution shift

![Distribution shift](figures/fig4_distribution_shift.png)
*Medians with whiskers to the 90th percentile over 200 fields (solvers and operators) or the minimum and maximum over five fields (PINNs).*

## Cost

Timed on a two-core CPU from \(a\) at the grid points to \(u\) at the grid points. Operator training includes generating the 1,000 training labels.

<!-- results:timing -->

![Cost of many queries](figures/fig5_cost.png)

!!! note "Timings are machine-dependent"
    The accuracy results are deterministic on the same hardware and software versions. Timings are not: they move between runs and machines, although their order did not change in any run made for this study.
