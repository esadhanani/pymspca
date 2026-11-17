"""Sparse PCA directly on a sparse TF-IDF matrix: interpretable "topics" with hard cardinality.

The covariance of the *centred* TF-IDF matrix is applied implicitly (rank-one centring
correction on the fly), so X is never densified and the p x p covariance is never formed.
For comparison: the dense centred matrix would need n * p * 8 bytes, which is what the msPCA
R package (dense X) and scikit-learn's SparsePCA (no sparse input) would have to allocate.

Run: python bench/text_topics.py
"""
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import pymspca

warnings.simplefilter("ignore", pymspca.MSPCAWarning)

docs = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).data
vec = TfidfVectorizer(min_df=5, max_df=0.5, stop_words="english", sublinear_tf=True, dtype=np.float64)
X = vec.fit_transform(docs)
vocab = np.array(vec.get_feature_names_out())
n, p = X.shape
print(f"TF-IDF: n={n:,} docs x p={p:,} features, nnz={X.nnz:,} ({X.nnz / (n * p):.3%} dense)")
print(f"dense centred X would need {n * p * 8 / 1e9:.1f} GB; dense covariance {p * p * 8 / 1e9:.1f} GB")

R, K = 8, 12
t0 = time.perf_counter()
res = pymspca.mspca(X, R, K, type="X", center=True, scale=False, constraint="orthogonality",
                    restarts="batched", max_restart_tpm=8, min_restart_tpm=2, max_iter=300, random_state=0)
dt = time.perf_counter() - t0
print(f"\npymspca: {R} components x {K} terms in {dt:.1f}s, {res.n_iter} iterations, "
      f"orthogonality violation {res.feasibility_violation:.1e}, FVE {res.fraction_variance_explained:.4f}\n")
for j in range(R):
    u = res.x_best[:, j]
    idx = np.flatnonzero(u)
    idx = idx[np.argsort(-np.abs(u[idx]))]
    print(f"PC{j + 1}: " + ", ".join(vocab[idx]))
