"""Bit-for-bit parity with the CRAN msPCA 0.5.1 R package, in deterministic mode (no random restarts).

Bit-exactness is a per-platform property: msPCA's heuristic amplifies last-bit differences in
floating-point summation order (which depends on the CPU's SIMD instructions) into different
local optima. So the reference outputs must come from R on the same platform as the test.

* Committed fixtures were produced on macOS arm64 by ``parity/run_r.R``; on other platforms
  this test is skipped.
* The ``r-parity`` CI job installs CRAN msPCA from source on each runner, regenerates the
  outputs with ``parity/run_r.R`` and points ``PYMSPCA_R_OUT`` at them.
* ``PYMSPCA_PARITY_EXACT=0`` relaxes bit-equality to: same support in every component,
  loadings and objective equal to 1e-10. This is used on Linux x86-64, where pymspca and
  R differ in the last bits.
"""
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymspca
from binio import read

FX = Path(__file__).parent / "fixtures"
CASES = pd.read_csv(FX / "cases.csv").to_dict("records")
R_OUT = Path(os.environ["PYMSPCA_R_OUT"]) if os.environ.get("PYMSPCA_R_OUT") else FX
FIXTURE_PLATFORM = ("Darwin", "arm64")
EXACT = os.environ.get("PYMSPCA_PARITY_EXACT", "1") != "0"
pytestmark = pytest.mark.skipif(
    R_OUT == FX and (platform.system(), platform.machine()) != FIXTURE_PLATFORM,
    reason="committed R fixtures are from macOS arm64; set PYMSPCA_R_OUT to outputs of parity/run_r.R on this platform")


@pytest.mark.parametrize("cs", CASES, ids=[f"{c['id']}-{c['file'][:-4]}-{c['type']}-c{c['ctype']}" for c in CASES])
def test_matches_r_exactly(cs):
    M = read(FX / (cs["file"][:-4] + ".bin"))
    ks = [int(k) for k in str(cs["ks"]).split(";")]
    with _quiet():
        res = pymspca.mspca(M, cs["r"], ks, type=cs["type"], constraint=int(cs["ctype"]), max_iter=int(cs["maxIter"]),
                            max_restart_tpm=0, min_restart_tpm=0, scale=str(cs["scale"]).upper() == "TRUE",
                            check_psd=False)
    U_r = read(R_OUT / f"{cs['id']}_U.bin")
    obj_r, viol_r, tot_r, _ = read(R_OUT / f"{cs['id']}_meta.bin").ravel()
    if EXACT:
        np.testing.assert_array_equal(res.x_best, U_r)
        assert res.objective_value == obj_r
        assert res.feasibility_violation == viol_r
    else:
        for j in range(U_r.shape[1]):
            assert set(np.flatnonzero(res.x_best[:, j])) == set(np.flatnonzero(U_r[:, j])), f"support of PC{j + 1}"
        np.testing.assert_allclose(res.x_best, U_r, rtol=0, atol=1e-10)
        assert res.objective_value == pytest.approx(obj_r, rel=1e-10)
    assert res.total_variance == pytest.approx(tot_r, rel=1e-14)


class _quiet:
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore", pymspca.MSPCAWarning)

    def __exit__(self, *a):
        return self._c.__exit__(*a)
