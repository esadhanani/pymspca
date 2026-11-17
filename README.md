# pymspca

Sparse PCA with multiple components, for Python. This is a port of the [msPCA](https://cran.r-project.org/package=msPCA) R package by Ryan Cory-Wright and Jean Pauphilet, which implements the method of

> R. Cory-Wright and J. Pauphilet. Sparse PCA with Multiple Components. *Operations Research*, 2026. [doi:10.1287/opre.2023.0598](https://doi.org/10.1287/opre.2023.0598)

Each component has exactly `k` non-zero loadings, and the components are either orthogonal or uncorrelated. The C++ core is msPCA's, with its R dependencies (random numbers, warnings, printing) passed in from Python, and it is compiled against the same Eigen version.

```python
import pymspca

res = pymspca.mspca(X, r=3, ks=10, type="X", constraint="uncorrelatedness")   # same arguments as msPCA::mspca
res.x_best                    # p x r loadings
res.feasibility_violation     # check this: above feasibility_tol means the constraint was not met

from pymspca import MultipleSparsePCA                                          # scikit-learn estimator
Z = MultipleSparsePCA(n_components=3, sparsity=10, random_state=0).fit_transform(X)
```

## Relation to the R package

- **Same results.** With random restarts off (`max_restart_tpm=0, min_restart_tpm=0`), pymspca reproduces msPCA 0.5.1 bit for bit on 18 reference cases, when both are built on the same platform. CI checks this on Linux x86-64 and macOS arm64 by installing CRAN msPCA from source on each runner ([parity/](parity/)). Across platforms the outputs can differ, as R's own do. With restarts on, results match R's statistically but not exactly, because the random number generators differ.
- **Sparse input.** `X` may be a `scipy.sparse` matrix. Centring and scaling are applied implicitly, as a rank-one correction in each matrix-vector product, so `X` is never densified and the p x p covariance is never formed.
- **Batched restarts** (`restarts="batched"`, opt-in). Each round of random restarts runs as fixed-width blocks of matrix-matrix products split across threads. For a given seed the result is the same for any number of threads.

Measured comparisons with scikit-learn's `SparsePCA` and the speed of batched restarts, including the cases where pymspca does worse, are in [BENCHMARKS.md](BENCHMARKS.md).

## Install

```bash
pip install git+https://github.com/esadhanani/pymspca
```

This needs a C++17 compiler; Eigen 3.4.0 is vendored. Run the tests with `PYTHONPATH=tests pytest`.

## Citing

Please cite the paper above ([CITATION.cff](CITATION.cff)).

## Licence and credits

MIT, as is msPCA. The algorithm and the original C++ are by Ryan Cory-Wright and Jean Pauphilet. The Python port is by Esa Dhanani, developed with AI assistance; the parity, sparse-input and threading results were checked by the author and are covered by the tests.
