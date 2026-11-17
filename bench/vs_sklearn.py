"""pymspca vs scikit-learn SparsePCA at matched sparsity.

sklearn's SparsePCA solves an l1-penalised dictionary-learning problem, so sparsity is set
indirectly through `alpha`. For a fair comparison we bisect `alpha` until sklearn's average
number of non-zeros per component matches msPCA's hard cardinality `k`, then evaluate both
loadings (columns normalised to unit length) on the same correlation matrix C:

  FVE          = tr(U^T C U) / tr(C)                          (higher is better)
  orth. viol.  = sum_{i<j} |u_i^T u_j|                        (0 = orthogonal)
  uncorr viol. = sum_{i<j} |u_i^T C u_j| / tr(C)              (0 = uncorrelated scores)

Run: python bench/vs_sklearn.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import SparsePCA

import pymspca

warnings.simplefilter("ignore")


def metrics(U, C):
    U = U / np.where(np.linalg.norm(U, axis=0) > 0, np.linalg.norm(U, axis=0), 1)
    tr = np.trace(C)
    i, j = np.triu_indices(U.shape[1], 1)
    return dict(FVE=np.trace(U.T @ C @ U) / tr, orth_viol=np.abs(U.T @ U)[i, j].sum(),
                uncorr_viol=np.abs(U.T @ C @ U)[i, j].sum() / tr, nnz=(U != 0).sum(0).mean())


def sklearn_matched(Xs, r, k, seed=0):
    lo, hi = 1e-4, 50.0
    best = None
    for _ in range(30):
        a = np.sqrt(lo * hi)
        m = SparsePCA(n_components=r, alpha=a, random_state=seed, max_iter=2000, tol=1e-8).fit(Xs)
        nnz = (m.components_ != 0).sum(1).mean()
        if best is None or abs(nnz - k) < abs(best[1] - k):
            best = (a, nnz, m)
        if abs(nnz - k) < 0.5:
            break
        lo, hi = (a, hi) if nnz > k else (lo, a)
    a = best[0]
    t0 = time.perf_counter()
    m = SparsePCA(n_components=r, alpha=a, random_state=seed, max_iter=2000, tol=1e-8).fit(Xs)
    return m.components_.T, time.perf_counter() - t0


def spiked(n, p, r, k, snr, seed):
    rng = np.random.default_rng(seed)
    U = np.zeros((p, r))
    for j in range(r):
        U[rng.choice(p, k, replace=False) if False else slice(j * k, (j + 1) * k), j] = rng.choice([-1, 1], k) / np.sqrt(k)
    return rng.standard_normal((n, r)) * np.sqrt(snr) @ U.T + rng.standard_normal((n, p))


DATA = {
    "breast cancer (n=569, p=30)": load_breast_cancer().data,
    "diabetes (n=442, p=10)": load_diabetes(scaled=False).data,
    "spiked (n=200, p=500)": spiked(200, 500, 5, 10, 4.0, 0),
}
CONFIG = {"breast cancer (n=569, p=30)": [(3, 5), (5, 5)], "diabetes (n=442, p=10)": [(3, 3)],
          "spiked (n=200, p=500)": [(5, 10)]}

rows = []
for name, X in DATA.items():
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    C = np.corrcoef(X, rowvar=False)
    for r, k in CONFIG[name]:
        for constraint in ("orthogonality", "uncorrelatedness"):
            t0 = time.perf_counter()
            res = pymspca.mspca(X, r, k, type="X", constraint=constraint, random_state=0)
            t = time.perf_counter() - t0
            rows.append(dict(data=name, r=r, k=k, method=f"pymspca ({constraint[:6]}.)", time_s=t, **metrics(res.x_best, C)))
        U, t = sklearn_matched(Xs, r, k)
        rows.append(dict(data=name, r=r, k=k, method="sklearn SparsePCA (alpha matched)", time_s=t, **metrics(U, C)))
        # Unconstrained dense PCA: an upper bound on FVE for any r components
        ev = np.linalg.eigvalsh(C)[::-1]
        rows.append(dict(data=name, r=r, k=k, method="dense PCA (upper bound, nnz=p)", time_s=np.nan,
                         FVE=ev[:r].sum() / ev.sum(), orth_viol=0.0, uncorr_viol=0.0, nnz=X.shape[1]))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
df.to_csv("bench/vs_sklearn_results.csv", index=False)
