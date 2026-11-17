"""Functional API mirroring the msPCA R package (``mspca()`` and ``tpm()``)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state

from . import _core

ArrayLike = Union[np.ndarray, "sp.spmatrix", "sp.sparray"]

_CONSTRAINTS = {"orthogonality": 0, "orthogonal": 0, 0: 0, "uncorrelatedness": 1, "uncorrelated": 1, 1: 1}


class MSPCAWarning(UserWarning):
    """Non-fatal warning raised by the msPCA solver (e.g. no feasible solution found)."""


@dataclass
class MSPCAResult:
    """Result of :func:`mspca`. Field names follow the R package."""

    x_best: np.ndarray                 # p x r loadings, columns sorted by explained variance
    objective_value: float             # tr(U^T Sigma U)
    feasibility_violation: float
    variance_explained: np.ndarray     # per-PC u_t^T Sigma u_t
    total_variance: float              # tr(Sigma)
    runtime: float
    n_iter: int
    constraint: str
    nonredundancy: dict = field(repr=False, default_factory=dict)
    preprocessing: dict = field(repr=False, default_factory=dict)

    @property
    def fraction_variance_explained(self) -> float:
        return float(self.objective_value / self.total_variance) if self.total_variance > 0 else float("nan")

    @property
    def fraction_variance_explained_per_pc(self) -> np.ndarray:
        return self.variance_explained / self.total_variance

    def __str__(self) -> str:
        r = self.x_best.shape[1]
        nnz = (self.x_best != 0).sum(axis=0)
        lines = [f"msPCA: {r} sparse PCs ({self.constraint}), {self.n_iter} iterations, {self.runtime:.3f}s",
                 f"  total fraction of variance explained: {self.fraction_variance_explained:.4f}",
                 f"  feasibility violation: {self.feasibility_violation:.2e}"]
        for t in range(r):
            lines.append(f"  PC{t + 1}: nnz={nnz[t]:>4d}  FVE={self.fraction_variance_explained_per_pc[t]:.4f}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- validation

def _constraint_code(c) -> int:
    key = c.lower() if isinstance(c, str) else c
    if key not in _CONSTRAINTS:
        raise ValueError("constraint must be 'orthogonality' or 'uncorrelatedness' (or 0 / 1)")
    return _CONSTRAINTS[key]


def _check_int(x, name, minimum=1, allow_zero=False):
    if isinstance(x, (bool, np.bool_)) or not float(x).is_integer():
        raise ValueError(f"{name} must be an integer, got {x!r}")
    x = int(x)
    lo = 0 if allow_zero else minimum
    if x < lo:
        raise ValueError(f"{name} must be >= {lo}, got {x}")
    return x


def _check_tol(x, name):
    x = float(x)
    if np.isnan(x) or x < 0:
        raise ValueError(f"{name} must be a non-negative number, got {x}")
    return min(x, np.finfo(float).max)


def _ks(ks, r, p) -> list:
    if np.isscalar(ks):
        ks = [ks] * r
    ks = [_check_int(k, "ks") for k in ks]
    return ks


def _seed(random_state) -> int:
    return int(check_random_state(random_state).randint(0, 2**63 - 1, dtype=np.int64))


def _emit(ws):
    for w in ws:
        warnings.warn(w, MSPCAWarning, stacklevel=3)


def _validate_sigma(S: np.ndarray, check_psd: bool, sym_tol: float, psd_tol: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 1:
        raise ValueError("Sigma must be a non-empty square matrix")
    if not np.all(np.isfinite(S)):
        raise ValueError("Sigma contains non-finite values (NaN or Inf)")
    if np.max(np.abs(S - S.T)) > sym_tol:
        raise ValueError("Sigma is not symmetric within sym_tol")
    if check_psd:
        lam = np.linalg.eigvalsh(S)[0]
        if lam < -psd_tol:
            raise ValueError(f"Sigma is not positive semidefinite (smallest eigenvalue {lam:.3e} < -psd_tol). "
                             "Pass a valid covariance/correlation matrix or set check_psd=False.")
    return S


# ----------------------------------------------------------------------------- preprocessing

def _check_divisor(divisor, n):
    div = str(divisor).lower()
    if div not in ("n-1", "n"):
        raise ValueError("divisor must be 'n-1' or 'n'")
    return 1.0 / (n - 1) if div == "n-1" else 1.0 / n


def _check_shape(X):
    n, p = X.shape
    if n < 2:
        raise ValueError("X must have at least 2 rows (observations)")
    if p < 1:
        raise ValueError("X must have at least 1 column (variable)")
    return n, p


def _zero_var_error(bad):
    raise ValueError("X has zero-variance column(s); cannot scale. Set scale=False or drop "
                     f"columns {np.flatnonzero(bad)[:10].tolist()}")


def scale_dense(X, center=True, scale=True):
    """Reproduce R's ``base::scale`` operation-for-operation (so results match bit-for-bit).

    R centres with ``colMeans`` (sequential sum, then divide), then divides each column by
    ``sqrt(sum(x^2) / (n - 1))`` of the *centred* column. The two-pass form also avoids the
    catastrophic cancellation of ``E[x^2] - E[x]^2``.
    Returns (Xproc, mean, sd).
    """
    n, p = _check_shape(X)
    X = np.ascontiguousarray(X, dtype=float)
    if center:
        mu = np.add.reduce(X, axis=0) / n  # row-wise sequential accumulation, like colMeans
        Xp = X - mu
    else:
        mu, Xp = np.zeros(p), X.copy()
    sd = np.ones(p)
    if scale:
        sd = np.sqrt(np.add.reduce(Xp * Xp, axis=0) / (n - 1))
        colmax = np.abs(X).max(axis=0)
        bad = sd <= np.sqrt(np.finfo(float).eps) * colmax
        if np.any(bad):
            _zero_var_error(bad)
        Xp = Xp / sd
    return Xp, mu, sd


def preprocess(X, center=True, scale=True, divisor="n-1"):
    """Return (mean, inv_scale, inv_divisor) for dense or scipy.sparse ``X`` without densifying it.

    For sparse ``X`` the centred sum of squares is accumulated stably as
    ``sum_nnz (x - mu)^2 + (n - nnz) * mu^2``.
    """
    n, p = _check_shape(X)
    inv_div = _check_divisor(divisor, n)
    if not sp.issparse(X):
        _, mu, sd = scale_dense(X, center, scale)
        return mu, 1.0 / sd, inv_div
    Xc = _as_csc(X)
    colsum = np.asarray(Xc.sum(axis=0)).ravel()
    mu = colsum / n if center else np.zeros(p)
    inv_sd = np.ones(p)
    if scale:
        nnz = np.diff(Xc.indptr)
        col = np.repeat(np.arange(p), nnz)
        dev = Xc.data - mu[col]
        ss = np.bincount(col, weights=dev * dev, minlength=p) + (n - nnz) * mu * mu
        sd = np.sqrt(ss / (n - 1))
        colmax = np.asarray(abs(Xc).max(axis=0).todense()).ravel()
        bad = sd <= np.sqrt(np.finfo(float).eps) * colmax
        if np.any(bad):
            _zero_var_error(bad)
        inv_sd = 1.0 / sd
    return mu, inv_sd, inv_div


def _as_csc(X):
    X = sp.csc_matrix(X, dtype=np.float64, copy=True)
    X.sum_duplicates()  # non-canonical CSC (repeated (i, j) entries) would corrupt the trace and scales
    if X.nnz >= 2**31:
        raise ValueError("sparse inputs with more than 2^31 - 1 non-zeros are not supported")
    X.sort_indices()
    return X


# ----------------------------------------------------------------------------- public functions

def mspca(M: ArrayLike, r: int, ks: Union[int, Sequence[int]], type: str = "Sigma",
          constraint: Union[str, int] = "orthogonality", verbose: bool = False, max_iter: int = 200,
          feasibility_tol: float = 1e-4, stalling_tol: float = 1e-8, time_limit_tpm: int = 20,
          max_restart_tpm: int = 30, min_restart_tpm: int = 20, center: bool = True, scale: bool = True,
          divisor: str = "n-1", check_psd: bool = True, sym_tol: float = 1e-8, psd_tol: float = 1e-8,
          random_state=None, restarts: str = "sequential") -> MSPCAResult:
    """Multiple sparse principal components (iterative deflation heuristic).

    Mirrors ``msPCA::mspca()``. ``type="Sigma"``: ``M`` is a p x p covariance or correlation
    matrix. ``type="X"``: ``M`` is an n x p data matrix (dense, or any ``scipy.sparse``
    matrix), centered/scaled exactly as R's ``base::scale`` and never materialised as a
    p x p matrix.

    ``restarts="sequential"`` (default) runs the random restarts of the truncated power
    method one at a time, as the R package does. With ``max_restart_tpm=0`` and
    ``min_restart_tpm=0`` the run is deterministic and reproduces R bit for bit on the same
    platform (the random restarts use a different generator from R's, so they cannot).
    ``restarts="batched"`` advances each round of restarts in fixed-width blocks of
    matrix-matrix products split across threads; results do not depend on the thread count.
    """
    t = str(type).lower()
    if t not in ("sigma", "x"):
        raise ValueError("type must be 'Sigma' or 'X'")
    r = _check_int(r, "r")
    ctype = _constraint_code(constraint)
    max_iter = _check_int(max_iter, "max_iter")
    time_limit_tpm = _check_int(time_limit_tpm, "time_limit_tpm", allow_zero=True)
    max_restart_tpm = _check_int(max_restart_tpm, "max_restart_tpm", allow_zero=True)
    min_restart_tpm = _check_int(min_restart_tpm, "min_restart_tpm", allow_zero=True)
    feasibility_tol, stalling_tol = _check_tol(feasibility_tol, "feasibility_tol"), _check_tol(stalling_tol, "stalling_tol")
    if not sp.issparse(M):
        M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("M must be 2-dimensional")
    p = M.shape[1]
    ks = _ks(ks, r, p)
    seed = _seed(random_state)
    if restarts not in ("sequential", "batched"):
        raise ValueError("restarts must be 'sequential' or 'batched'")
    args = (r, ks, ctype, bool(verbose), max_iter, feasibility_tol, stalling_tol, time_limit_tpm,
            max_restart_tpm, min_restart_tpm, seed, restarts == "batched")
    pre = {"type": "Sigma" if t == "sigma" else "X"}

    if t == "sigma":
        if sp.issparse(M):
            M = M.toarray()
        S = _validate_sigma(M, check_psd, sym_tol, psd_tol)
        res = _core.mspca_sigma(np.asfortranarray(S), *args)
        U = res["x_best"]
        SU = S @ U
    else:
        if sp.issparse(M):
            Xc = _as_csc(M)
            if not np.all(np.isfinite(Xc.data)):
                raise ValueError("X contains non-finite values (NaN or Inf)")
            mu, inv_sd, inv_div = preprocess(Xc, center, scale, divisor)
            res = _core.mspca_sparse(Xc.shape[0], Xc.shape[1], Xc.data, Xc.indices, Xc.indptr, mu, inv_sd, inv_div, *args)
            U = res["x_best"]
            W = U * inv_sd[:, None]
            XU = Xc @ W - (mu @ W)[None, :]
            SU = inv_div * (inv_sd[:, None] * (Xc.T @ XU - np.outer(mu, XU.sum(axis=0))))
        else:
            X = np.asarray(M, dtype=float)
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            if not np.all(np.isfinite(X)):
                raise ValueError("X contains non-finite values (NaN or Inf)")
            Xp, mu, sd = scale_dense(X, center, scale)
            inv_sd, inv_div = 1.0 / sd, _check_divisor(divisor, X.shape[0])
            Xp = np.asfortranarray(Xp)
            res = _core.mspca_x(Xp, inv_div, *args)
            U = res["x_best"]
            SU = inv_div * (Xp.T @ (Xp @ U))
        pre.update(mean=mu, inv_scale=inv_sd, inv_divisor=inv_div, center=center, scale=scale, divisor=divisor,
                   n_obs=M.shape[0], sparse_input=sp.issparse(M))

    _emit(res["warnings"])
    tr = res["total_variance"] if res["total_variance"] > 0 else 1.0
    upper = np.triu(np.ones((U.shape[1],) * 2, dtype=bool), 1)
    ortho, uncor = np.abs(U.T @ U), np.abs(U.T @ SU) / tr
    ortho[~upper] = np.nan
    uncor[~upper] = np.nan
    return MSPCAResult(x_best=U, objective_value=res["objective_value"],
                       feasibility_violation=res["feasibility_violation"],
                       variance_explained=np.asarray(res["variance_explained"]),
                       total_variance=res["total_variance"], runtime=res["runtime"], n_iter=res["n_iter"],
                       constraint="orthogonality" if ctype == 0 else "uncorrelatedness",
                       nonredundancy={"orthogonality": ortho, "uncorrelatedness": uncor}, preprocessing=pre)


@dataclass
class TPMResult:
    x_best: np.ndarray
    objective_value: float
    runtime: float


def tpm(M: ArrayLike, k: int, type: str = "Sigma", max_iter: int = 200, verbose: bool = False,
        time_limit: int = 10, center: bool = True, scale: bool = True, divisor: str = "n-1",
        check_psd: bool = True, sym_tol: float = 1e-8, psd_tol: float = 1e-8, random_state=None) -> TPMResult:
    """Single sparse PC via the truncated power method (mirrors ``msPCA::tpm()``)."""
    k = _check_int(k, "k")
    if not sp.issparse(M):
        M = np.asarray(M, dtype=float)
    max_iter = _check_int(max_iter, "max_iter", allow_zero=True)
    time_limit = _check_int(time_limit, "time_limit", allow_zero=True)
    seed = _seed(random_state)
    t = str(type).lower()
    if t == "sigma":
        S = _validate_sigma(M.toarray() if sp.issparse(M) else M, check_psd, sym_tol, psd_tol)
        res = _core.tpm_sigma(np.asfortranarray(S), k, max_iter, bool(verbose), time_limit, seed)
    elif t == "x":
        if sp.issparse(M):
            Xc = _as_csc(M)
            mu, inv_sd, inv_div = preprocess(Xc, center, scale, divisor)
            res = _core.tpm_sparse(Xc.shape[0], Xc.shape[1], Xc.data, Xc.indices, Xc.indptr, mu, inv_sd, inv_div,
                                   k, max_iter, bool(verbose), time_limit, seed)
        else:
            X = np.asarray(M, dtype=float)
            Xp, _, _ = scale_dense(X, center, scale)
            res = _core.tpm_x(np.asfortranarray(Xp), _check_divisor(divisor, X.shape[0]), k, max_iter, bool(verbose),
                              time_limit, seed)
    else:
        raise ValueError("type must be 'Sigma' or 'X'")
    _emit(res["warnings"])
    return TPMResult(x_best=res["x_best"], objective_value=res["objective_value"], runtime=res["runtime"])
