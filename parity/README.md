# Parity with the R package

`tests/test_parity_r.py` runs 18 deterministic cases (no random restarts). They cover both constraints, `type="Sigma"` and `type="X"`, and scaled and unscaled data, on mtcars, msPCA's S&P 500 correlation matrix (p = 423) and spiked models up to p = 2000. The test requires `max |ΔU| = 0`, identical objective and identical feasibility violation.

Regenerate the R reference on your machine:

```bash
Rscript -e 'install.packages("msPCA", type = "source")'
Rscript parity/run_r.R r_out          # writes r_out/<case>_U.bin, r_out/<case>_meta.bin
PYMSPCA_R_OUT=r_out PYTHONPATH=tests pytest tests/test_parity_r.py
```

The committed fixtures in `tests/fixtures/` come from macOS arm64, and the test skips itself on other platforms unless `PYMSPCA_R_OUT` is set.

The `r-parity` CI job installs CRAN msPCA from source on each runner and regenerates the reference. Results:

| platform | bit-identical | same support |
|---|---|---|
| macOS arm64 | 18/18 | 18/18 |
| Linux x86-64 | 2/18 | 18/18 |

On Linux the job asserts identical supports and objectives to 1e-3, and prints the full comparison (`parity/compare.py`). The objectives agree to 1e-10 in 15 of the 18 cases. The worst three are unscaled or small mtcars and one spiked case, with relative objective differences of 3e-5, 2e-4 and 1e-6.

## Why it is per-platform

The heuristic is sensitive to floating-point summation order. A last-bit difference in a matrix product can send it to a different local optimum. Bit-exact agreement therefore needs the same arithmetic in the same order:

1. The same Eigen, 3.4.0, which is what RcppEigen bundles. A different Eigen gives different results.
2. Centring and scaling done the way R's `base::scale` does them: centre in a separate pass, then divide by the scale. Multiplying by its reciprocal gives different bits.
3. The data matrix in `GramOp` held as an owning `Eigen::MatrixXd`. A strided `Eigen::Ref` view makes Eigen choose a different matrix-product kernel.
4. Inputs and outputs exchanged with R as raw float64 (`binio.py`). R's CSV parser does not round 17-digit decimals correctly.

The same applies across CPUs. NEON on arm64 uses fused multiply-adds, while SSE2 on x86-64 does not, and R accumulates column means in 80-bit `long double` on x86. On Linux, pymspca and R agree only up to the last few bits. Rebuilding pymspca on macOS with only `-DEIGEN_DONT_VECTORIZE` moves 8 of the 18 cases to a different support.
