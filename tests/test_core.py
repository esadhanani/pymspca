import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import pymspca
from pymspca import MultipleSparsePCA, mspca, tpm

warnings.simplefilter("ignore", pymspca.MSPCAWarning)


def spiked(n=150, p=80, r=3, k=6, snr=5.0, seed=0):
    rng = np.random.default_rng(seed)
    U = np.zeros((p, r))
    for j in range(r):
        U[j * k:(j + 1) * k, j] = rng.choice([-1, 1], k) / np.sqrt(k)
    return rng.standard_normal((n, r)) * np.sqrt(snr) @ U.T + rng.standard_normal((n, p)), U


# ---------------------------------------------------------------- correctness properties

@pytest.mark.parametrize("constraint", ["orthogonality", "uncorrelatedness"])
def test_cardinality_and_feasibility(constraint):
    X, _ = spiked()
    res = mspca(X, 3, [6, 5, 4], type="X", constraint=constraint, random_state=0)
    nnz = sorted((res.x_best != 0).sum(axis=0).tolist())
    assert nnz == [4, 5, 6]
    np.testing.assert_allclose(np.linalg.norm(res.x_best, axis=0), 1.0, atol=1e-12)
    assert res.feasibility_violation < 1e-4
    assert np.all(np.diff(res.variance_explained) <= 1e-12)  # sorted by explained variance


def test_recovers_planted_supports():
    X, U = spiked(n=400, p=120, r=3, k=8, snr=8.0, seed=3)
    res = mspca(X, 3, 8, type="X", random_state=1)
    true = [set(np.flatnonzero(U[:, j])) for j in range(3)]
    found = [set(np.flatnonzero(res.x_best[:, j])) for j in range(3)]
    for t in true:
        assert t in found


def test_sigma_and_x_paths_agree():
    X, _ = spiked()
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    S = np.cov(Xs, rowvar=False)
    a = mspca(S, 3, 6, type="Sigma", max_restart_tpm=0, min_restart_tpm=0)
    b = mspca(X, 3, 6, type="X", max_restart_tpm=0, min_restart_tpm=0)
    np.testing.assert_allclose(a.objective_value, b.objective_value, rtol=1e-8)
    np.testing.assert_allclose(np.abs(a.x_best), np.abs(b.x_best), atol=1e-7)


def test_total_variance_matches_trace():
    X, _ = spiked()
    for scale in (True, False):
        res = mspca(X, 2, 5, type="X", scale=scale, random_state=0)
        C = np.corrcoef(X, rowvar=False) if scale else np.cov(X, rowvar=False)
        assert res.total_variance == pytest.approx(np.trace(C), rel=1e-12)
        np.testing.assert_allclose(res.variance_explained,
                                   np.einsum("pi,pq,qi->i", res.x_best, C, res.x_best), rtol=1e-10)


def test_nonredundancy_matrices():
    X, _ = spiked()
    res = mspca(X, 3, 6, type="X", constraint="uncorrelatedness", random_state=0)
    C = np.corrcoef(X, rowvar=False)
    U = res.x_best
    nr = res.nonredundancy
    i, j = np.triu_indices(3, 1)
    np.testing.assert_allclose(nr["orthogonality"][i, j], np.abs(U.T @ U)[i, j], atol=1e-12)
    np.testing.assert_allclose(nr["uncorrelatedness"][i, j], (np.abs(U.T @ C @ U) / np.trace(C))[i, j], atol=1e-12)
    assert np.isnan(nr["orthogonality"][np.tril_indices(3)]).all()


# ---------------------------------------------------------------- sparse input path (new)

@pytest.mark.parametrize("center,scale", [(True, True), (True, False), (False, False), (False, True)])
@pytest.mark.parametrize("constraint", [0, 1])
def test_sparse_matches_dense(center, scale, constraint):
    rng = np.random.default_rng(7)
    X = sp.random(300, 150, density=0.05, random_state=rng, data_rvs=lambda n: rng.standard_normal(n) + 2).tocsr()
    X[:, :6] = X[:, :6] + sp.csr_matrix(np.outer(rng.standard_normal(300), np.ones(6)) * (rng.random((300, 6)) < 0.3))
    kw = dict(r=3, ks=6, type="X", constraint=constraint, center=center, scale=scale,
              max_restart_tpm=0, min_restart_tpm=0)
    d = mspca(X.toarray(), **kw)
    s = mspca(X, **kw)
    assert s.total_variance == pytest.approx(d.total_variance, rel=1e-10)
    # The reported objective of the sparse run is tr(U^T C U) for the explicitly formed C.
    Xd = X.toarray()
    Xp = (Xd - Xd.mean(0)) if center else Xd
    if scale:
        Xp = Xp / np.sqrt((Xp ** 2).sum(0) / (Xd.shape[0] - 1))
    C = Xp.T @ Xp / (Xd.shape[0] - 1)
    assert s.objective_value == pytest.approx(np.trace(s.x_best.T @ C @ s.x_best), rel=1e-9)
    if constraint == 0:
        np.testing.assert_allclose(np.abs(s.x_best), np.abs(d.x_best), atol=1e-6)
    else:
        # Uncorrelatedness is sensitive to summation order: sparse and dense products may
        # reach different local optima, so compare quality rather than loadings.
        assert s.objective_value == pytest.approx(d.objective_value, rel=0.05)


def test_sparse_never_densifies():
    # 20k x 200k at 0.01% density: dense would need 32 GB.
    rng = np.random.default_rng(0)
    X = sp.random(20_000, 200_000, density=1e-4, format="csr", random_state=rng)
    res = mspca(X, 2, 10, type="X", scale=False, max_iter=20, random_state=0)
    assert res.x_best.shape == (200_000, 2)
    assert ((res.x_best != 0).sum(0) == 10).all()


def test_sparse_zero_variance_column_rejected():
    X = sp.csr_matrix(np.c_[np.random.default_rng(0).standard_normal((20, 3)), np.zeros(20)])
    with pytest.raises(ValueError, match="zero-variance"):
        mspca(X, 1, 2, type="X", scale=True)


# ---------------------------------------------------------------- tpm

def test_tpm_single_component():
    X, U = spiked(r=1, k=6, snr=10.0)
    S = np.corrcoef(X, rowvar=False)
    res = tpm(S, 6, random_state=0)
    assert set(np.flatnonzero(res.x_best[:, 0])) == set(np.flatnonzero(U[:, 0]))
    a = tpm(X, 6, type="X", random_state=0)
    b = tpm(sp.csr_matrix(X), 6, type="X", random_state=0)
    assert a.objective_value == pytest.approx(b.objective_value, rel=1e-8)


# ---------------------------------------------------------------- reproducibility

def test_random_state_reproducible():
    X, _ = spiked()
    a = mspca(X, 3, 6, type="X", random_state=42)
    b = mspca(X, 3, 6, type="X", random_state=42)
    np.testing.assert_array_equal(a.x_best, b.x_best)


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("bad,match", [
    (np.array([[1.0, 2.0], [0.0, 1.0]]), "symmetric"),
    (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    (np.array([[np.nan, 0], [0, 1.0]]), "non-finite"),
    (np.ones((2, 3)), "square"),
])
def test_sigma_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        mspca(bad, 1, 1)


def test_argument_validation():
    S = np.eye(4)
    with pytest.raises(ValueError):
        mspca(S, 0, 1)
    with pytest.raises(ValueError):
        mspca(S, 1, 1.5)
    with pytest.raises(ValueError):
        mspca(S, 1, 1, constraint="sideways")
    with pytest.raises(ValueError):
        mspca(S, 1, 1, type="Z")
    with pytest.raises(ValueError):
        mspca(np.ones((1, 3)), 1, 1, type="X")


def test_warns_when_infeasible():
    X, _ = spiked()
    with pytest.warns(pymspca.MSPCAWarning, match="feasible"):
        mspca(X, 3, 6, type="X", max_iter=1)


def test_sparsity_above_dimension_warns():
    with pytest.warns(pymspca.MSPCAWarning, match="exceeds the dimension"):
        mspca(np.eye(3), 1, 5)


# ---------------------------------------------------------------- scikit-learn estimator

def test_estimator_transform_roundtrip():
    X, _ = spiked()
    est = MultipleSparsePCA(n_components=3, sparsity=6, random_state=0).fit(X)
    Z = est.transform(X)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    np.testing.assert_allclose(Z, Xs @ est.loadings_, atol=1e-10)
    np.testing.assert_allclose(est.transform(sp.csr_matrix(X)), Z, atol=1e-10)
    assert est.components_.shape == (3, X.shape[1])
    assert est.get_feature_names_out().tolist() == ["multiplesparsepca0", "multiplesparsepca1", "multiplesparsepca2"]
    assert est.inverse_transform(Z).shape == X.shape


def test_estimator_in_pipeline():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    X, U = spiked(n=200)
    y = X @ U[:, 0] + 0.1 * np.random.default_rng(0).standard_normal(200)  # target driven by planted factor 1
    pipe = make_pipeline(MultipleSparsePCA(3, 6, random_state=0, max_restart_tpm=5, min_restart_tpm=2),
                         LinearRegression())
    assert cross_val_score(pipe, X, y, cv=3).mean() > 0.5


def test_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    est = MultipleSparsePCA(n_components=1, sparsity=2, max_iter=10, max_restart_tpm=2, min_restart_tpm=1,
                            random_state=0, scale=False)
    check_estimator(est)


# ---------------------------------------------------------------- batched / threaded restarts

@pytest.mark.parametrize("kind", ["sigma", "x", "sparse"])
def test_batched_results_independent_of_thread_count(kind):
    # Thread counts chosen to hit every remainder pattern of the restart panels.
    X, _ = spiked(n=120, p=420, r=3, k=8, seed=5)
    if kind == "sigma":
        M, t = np.corrcoef(X, rowvar=False), "Sigma"
    elif kind == "sparse":
        X[np.abs(X) < 1.0] = 0.0
        M, t = sp.csr_matrix(X), "X"
    else:
        M, t = X, "X"
    out = {}
    try:
        for nt in (1, 2, 3, 7, 13, 20, 30):
            pymspca.set_num_threads(nt)
            out[nt] = mspca(M, 3, 8, type=t, restarts="batched", random_state=11, max_iter=40).x_best
    finally:
        pymspca.set_num_threads(0)
    for nt, U in out.items():
        np.testing.assert_array_equal(U, out[1], err_msg=f"{nt} threads differ from 1 thread")


def test_estimator_default_is_reproducible_sequential():
    assert MultipleSparsePCA().restarts == "sequential"


def test_sparse_duplicate_entries_are_summed():
    rng = np.random.default_rng(3)
    D = rng.standard_normal((60, 25)) * (rng.random((60, 25)) < 0.3)
    A = sp.csc_matrix(D)
    # same matrix in non-canonical CSC: every value split into two duplicate (i, j) entries
    cnt = np.diff(A.indptr)
    data = np.concatenate([np.r_[A.data[s:e] * 0.25, A.data[s:e] * 0.75] for s, e in zip(A.indptr[:-1], A.indptr[1:])])
    rows = np.concatenate([np.r_[A.indices[s:e], A.indices[s:e]] for s, e in zip(A.indptr[:-1], A.indptr[1:])])
    dup = sp.csc_matrix((data, rows, np.r_[0, np.cumsum(2 * cnt)]), shape=A.shape)
    assert dup.nnz == 2 * A.nnz and not dup.has_canonical_format
    ref = mspca(sp.csc_matrix(D), 2, 4, type="X", max_restart_tpm=0, min_restart_tpm=0)
    got = mspca(dup, 2, 4, type="X", max_restart_tpm=0, min_restart_tpm=0)
    assert got.total_variance == pytest.approx(ref.total_variance, rel=1e-12)
    np.testing.assert_allclose(got.x_best, ref.x_best, atol=1e-10)


def test_sparse_trace_stable_with_large_means():
    rng = np.random.default_rng(4)
    n, p = 400, 12
    D = 1e8 + rng.standard_normal((n, p))  # mean 1e8, sd 1: the one-pass trace formula is off by ~6% here
    dense = mspca(D, 1, 3, type="X", scale=False, max_restart_tpm=0, min_restart_tpm=0)
    sparse = mspca(sp.csc_matrix(D), 1, 3, type="X", scale=False, max_restart_tpm=0, min_restart_tpm=0)
    assert sparse.total_variance == pytest.approx(dense.total_variance, rel=1e-9)


def test_accepts_nested_lists():
    X, _ = spiked(n=40, p=20)
    a = mspca(X.tolist(), 2, 3, type="X", random_state=0)
    b = mspca(X, 2, 3, type="X", random_state=0)
    np.testing.assert_array_equal(a.x_best, b.x_best)


@pytest.mark.parametrize("constraint", [0, 1])
def test_batched_matches_sequential_quality(constraint):
    X, U = spiked(n=300, p=150, r=3, k=8, snr=6.0, seed=9)
    seq = mspca(X, 3, 8, type="X", constraint=constraint, random_state=0)
    bat = mspca(X, 3, 8, type="X", constraint=constraint, random_state=0, restarts="batched")
    assert bat.fraction_variance_explained >= 0.98 * seq.fraction_variance_explained
    assert bat.feasibility_violation < 1e-3
