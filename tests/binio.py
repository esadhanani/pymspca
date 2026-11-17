"""Exact float64 exchange with R (CSV text round-trips are not bit-exact in R's strtod)."""
import numpy as np


def write(path, A):
    A = np.asarray(A, dtype="<f8")
    with open(path, "wb") as f:
        np.array(A.shape, dtype="<i4").tofile(f)
        A.ravel(order="F").tofile(f)


def read(path):
    with open(path, "rb") as f:
        n, p = np.fromfile(f, dtype="<i4", count=2)
        return np.fromfile(f, dtype="<f8").reshape((n, p), order="F")
