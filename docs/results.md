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

Every method receives the same NumPy array holding \(a\) at the grid points and returns a NumPy array with \(u\) at the grid points; conversions and preprocessing are inside the timed region. All times, training included, were measured by `timing.py` on one two-core CPU. Operator training includes generating the training labels.

<!-- results:timing -->

![Cost of many queries](figures/fig5_cost.png)

!!! note "Timings are machine-dependent"
    The accuracy results are reproducible bit for bit on the same CPU model and software versions; on other hardware the network-based numbers can differ in the last digits. Timings vary between runs and machines, although the order of the methods did not change in any run made for this study.
