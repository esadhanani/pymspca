# Benchmarks

All runs used an Apple M1 Max. Scripts and raw CSVs are in [bench/](bench/). FVE is the fraction of variance explained, tr(UᵀCU)/tr(C), the paper's objective. When scores are correlated it double-counts variance, so read it together with the uncorrelatedness column. A violation above the default `feasibility_tol` of 1e-4 means the constraint was not met. Those runs are marked *infeasible*.

## Against scikit-learn's `SparsePCA` at matched sparsity

Script: [bench/vs_sklearn.py](bench/vs_sklearn.py). Raw results: [bench/vs_sklearn_results.csv](bench/vs_sklearn_results.csv).

sklearn controls sparsity through an l1 penalty `alpha`. The script bisects `alpha` until sklearn's mean number of non-zeros per component matches `k`, then scores both methods on the same correlation matrix. The two right-hand columns measure how redundant the components are:

- Σ|uᵢᵀuⱼ| is the overlap between loading vectors (0 means orthogonal).
- Σ|uᵢᵀCuⱼ|/tr C is the correlation between component scores (0 means uncorrelated).

| data | r × k | method | FVE | Σ\|uᵢᵀuⱼ\| | Σ\|uᵢᵀCuⱼ\|/tr C | nnz/PC | time (s) |
|---|---|---|---|---|---|---|---|
| breast cancer (p=30) | 3 × 5 | pymspca, orthogonal | 0.430 | 9e-5 | 0.32 | 5 | 3.2 |
| | | pymspca, uncorrelated | 0.320 | 0 | 9e-5 | 5 | 4.0 |
| | | sklearn SparsePCA | 0.419 | 0 | 0.12 | 5 | 0.03 |
| | | dense PCA | 0.726 | 0 | 0 | 30 | |
| breast cancer (p=30) | 5 × 5 | pymspca, orthogonal | 0.612 | 9e-5 | 0.66 | 5 | 7.2 |
| | | pymspca, uncorrelated | 0.385 | 0.60 | 4.4e-4 *(infeasible)* | 5 | 7.4 |
| | | sklearn SparsePCA | 0.609 | 0.03 | 0.45 | 4.6 | 0.04 |
| | | dense PCA | 0.847 | 0 | 0 | 30 | |
| diabetes (p=10) | 3 × 3 | pymspca, orthogonal | 0.596 | 0 | 0.28 | 3 | 0.2 |
| | | pymspca, uncorrelated | 0.476 | 0.40 | 1e-4 | 3 | 1.2 |
| | | sklearn SparsePCA | 0.598 | 0 | 0.23 | 3 | 0.01 |
| | | dense PCA | 0.672 | 0 | 0 | 10 | |
| spiked (n=200, p=500) | 5 × 10 | pymspca, orthogonal | 0.0347 | 0 | 0.005 | 10 | 4.2 |
| | | pymspca, uncorrelated | 0.0219 | 0.13 | 4e-5 | 10 | 54.6 |
| | | sklearn SparsePCA | 0.0347 | 0 | 0.005 | 10.2 | 0.16 |
| | | dense PCA | 0.0698 | 0 | 0 | 500 | |

What the table shows:
- **Variance explained.** In orthogonal mode, pymspca and sklearn explain about the same variance: pymspca is slightly ahead on breast cancer and level or slightly behind elsewhere.
- **Exact sparsity.** pymspca always has exactly `k` non-zeros per component. sklearn's l1 penalty lands near `k` but not on it (4.6 and 10.2 above), and reaching even that took 10–30 extra fits per configuration to tune `alpha`.
- **Uncorrelated scores.** sklearn has no way to make component scores uncorrelated. pymspca's uncorrelatedness mode does, at a clear cost in variance explained. It also missed its tolerance once (breast cancer 5 × 5).
- **Speed.** sklearn is 10–300× faster per fit, even before its extra `alpha` fits.

## Restarts: sequential vs batched

Script: [bench/restarts_speed.py](bench/restarts_speed.py). Raw results: [bench/restarts_speed_results.csv](bench/restarts_speed_results.csv).

This uses the default restart budget (30 maximum, 20 minimum), 3 seeds per configuration, and 10 threads. "Worst violation" is the largest over the 3 seeds.

| problem | constraint | sequential (as in R) | batched, 1 thread | batched, 10 threads | mean FVE, sequential → batched | worst violation, seq. / batched |
|---|---|---|---|---|---|---|
| breast cancer Σ, p=30, 5 × 5 | orth. | 0.77 s | 0.58 s (1.3×) | 0.58 s (1.3×) | 0.597 → 0.616 | 9e-5 / 1e-4 |
| | uncorr. | 0.82 s | 0.63 s (1.3×) | 0.63 s (1.3×) | 0.389 → 0.386 | 4.4e-4 / 1.4e-3 *(both infeasible)* |
| S&P 500 Σ, p=423, 5 × 10 | orth. | 47.4 s | 39.1 s (1.2×) | 16.9 s (2.8×) | 0.0682 → 0.0683 | 1e-4 / 1e-4 |
| | uncorr. | 46.6 s | 35.7 s (1.3×) | 15.4 s (3.0×) | 0.0433 → 0.0433 | 1e-4 / 1e-4 |
| spiked X, n=80, p=2000, 4 × 15 | orth. | 3.7 s | 3.3 s (1.1×) | 1.4 s (2.7×) | 0.0114 → 0.0113 | 8e-16 / 6e-16 |
| | uncorr. | 82.8 s | 72.5 s (1.1×) | 30.8 s (2.7×) | 0.0054 → 0.0053 | 1e-4 / 8e-5 |

What the table shows:
- **Same quality.** Batched restarts reach the same solution quality as sequential, within the spread across seeds.
- **Where the speed comes from.** Batching alone gives only 1.1–1.3×. Most of the gain comes from running whole blocks on separate threads, so a 2-core laptop will see much less.
- **Thread-count invariance.** The 1-thread and 10-thread results are identical for every seed, as the tests require.
- **Constraint not always met.** On breast cancer, uncorrelatedness with the default settings misses the 1e-4 tolerance in both modes.

## A case it does not solve: sparse text

Script: [bench/text_topics.py](bench/text_topics.py). Log: [bench/text_topics.log](bench/text_topics.log).

Data: 20 Newsgroups TF-IDF, n = 11,314 documents, p = 17,797 terms, 0.3% non-zero. Run: r = 8 components of k = 12 terms each, 300 iterations, reduced restarts (8/2), no scaling.

- **Sparse input worked.** Peak memory was 0.35 GB. A dense centred X would need 1.6 GB, and the covariance 2.5 GB.
- **The constraint was not met.** The orthogonality violation was still 0.78 after 393 s. Three of the eight components are near-duplicates built from generic words ("thanks", "know", "advance", "help"). Some components are clean, for example drive / scsi / hard / disk / controller / ide, but as a whole this run is infeasible and is not a usable result.

## S&P 500 sparse factors

Script: [bench/snp500_factors.py](bench/snp500_factors.py). Results: [bench/snp500_factors_results.csv](bench/snp500_factors_results.csv).

Data: msPCA's `snp500` case-study matrix, the market-deflated correlations of 423 constituents, 2010–2019. Run: 8 orthogonal factors of 10 names each.

The leading factors are recognisable industry baskets:
- **PC1, utilities:** ED, DUK, WEC, XEL, AEP…
- **PC2, REITs:** WELL, AVB, UDR, ESS, EQR…
- **PC3, oil and gas:** DVN, APA, COP, EOG, HAL…

**Sector purity** is the share of a factor's names that fall in its most common GICS sector. It averages 78% here, against 77% for the 10 largest-loading names of each dense principal component. So the gain is a factor you can read without thresholding loadings, not purer sectors.
