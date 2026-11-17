"""scikit-learn estimator wrapper."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._api import mspca


class MultipleSparsePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Sparse PCA with multiple, non-redundant components.

    Implements the iterative deflation heuristic of Cory-Wright & Pauphilet,
    "Sparse PCA With Multiple Components", *Operations Research* 74(5), 2026,
    via the same C++ core as the ``msPCA`` R package. Unlike
    :class:`sklearn.decomposition.SparsePCA` (an l1-penalised dictionary-learning
    formulation), sparsity is a hard cardinality constraint (exactly ``k`` non-zero
    loadings per component) and components are explicitly constrained to be
    orthogonal or uncorrelated.

    Parameters
    ----------
    n_components : int
    sparsity : int or sequence of int
        Number of non-zero loadings per component (``ks`` in the R package).
    constraint : {"orthogonality", "uncorrelatedness"}
    center, scale : bool
        Center / standardise columns (``scale=True`` targets the correlation matrix,
        the R default).
    precomputed : bool
        If True, ``fit`` expects a p x p covariance/correlation matrix instead of data.
    restarts : {"sequential", "batched"}
        ``"sequential"`` (default) runs the random restarts one at a time, as the R package
        does. ``"batched"`` advances each round of restarts as blocks of matrix-matrix
        products split across threads; it is faster on larger problems, and its results
        do not depend on the number of threads.
    random_state : int, RandomState or None
        Seeds the random restarts of the truncated power method.

    Attributes
    ----------
    components_ : ndarray (n_components, n_features)
    loadings_ : ndarray (n_features, n_components), R's ``x_best``
    explained_variance_, explained_variance_ratio_ : ndarray (n_components,)
    total_variance_, feasibility_violation_, n_iter_, runtime_
    nonredundancy_ : dict of pairwise |u_t' u_s| and |u_t' Sigma u_s| / tr(Sigma)
    mean_, scale_ : ndarray (n_features,) (data input only)
    """

    def __init__(self, n_components=2, sparsity=10, constraint="orthogonality", center=True, scale=True,
                 divisor="n-1", precomputed=False, max_iter=200, feasibility_tol=1e-4, stalling_tol=1e-8,
                 max_restart_tpm=30, min_restart_tpm=20, time_limit_tpm=20, restarts="sequential", random_state=None, verbose=False):
        self.n_components = n_components
        self.sparsity = sparsity
        self.constraint = constraint
        self.center = center
        self.scale = scale
        self.divisor = divisor
        self.precomputed = precomputed
        self.max_iter = max_iter
        self.feasibility_tol = feasibility_tol
        self.stalling_tol = stalling_tol
        self.max_restart_tpm = max_restart_tpm
        self.min_restart_tpm = min_restart_tpm
        self.time_limit_tpm = time_limit_tpm
        self.restarts = restarts
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        X = validate_data(self, X, accept_sparse=["csr", "csc", "coo"], dtype=np.float64,
                          ensure_min_samples=1 if self.precomputed else 2)
        res = mspca(X, r=self.n_components, ks=self.sparsity, type="Sigma" if self.precomputed else "X",
                    constraint=self.constraint, verbose=self.verbose, max_iter=self.max_iter,
                    feasibility_tol=self.feasibility_tol, stalling_tol=self.stalling_tol,
                    time_limit_tpm=self.time_limit_tpm, max_restart_tpm=self.max_restart_tpm,
                    min_restart_tpm=self.min_restart_tpm, center=self.center, scale=self.scale,
                    divisor=self.divisor, random_state=self.random_state, restarts=self.restarts)
        self.result_ = res
        self.loadings_ = res.x_best
        self.components_ = res.x_best.T.copy()
        self.n_components_ = self._n_features_out = res.x_best.shape[1]
        self.explained_variance_ = res.variance_explained
        self.explained_variance_ratio_ = res.fraction_variance_explained_per_pc
        self.total_variance_ = res.total_variance
        self.feasibility_violation_ = res.feasibility_violation
        self.nonredundancy_ = res.nonredundancy
        self.n_iter_ = res.n_iter
        self.runtime_ = res.runtime
        if not self.precomputed:
            self.mean_ = res.preprocessing["mean"]
            self.scale_ = 1.0 / res.preprocessing["inv_scale"]
        return self

    def transform(self, X):
        """Scores: the centred/scaled data projected on the sparse loadings."""
        check_is_fitted(self, "loadings_")
        if self.precomputed:
            raise ValueError("transform is unavailable when the model was fitted on a covariance matrix")
        X = validate_data(self, X, accept_sparse=["csr", "csc"], dtype=np.float64, reset=False)
        W = self.loadings_ / self.scale_[:, None]
        Z = X @ W - (self.mean_ @ W)[None, :]
        return np.asarray(Z)

    def inverse_transform(self, Z):
        check_is_fitted(self, "loadings_")
        if self.precomputed:
            raise ValueError("inverse_transform is unavailable when the model was fitted on a covariance matrix")
        # Least-squares reconstruction: loadings are only approximately orthonormal
        # (exactly so only up to feasibility_tol under the orthogonality constraint).
        return (np.asarray(Z) @ np.linalg.pinv(self.loadings_)) * self.scale_ + self.mean_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags
