"""Run every parity case through pymspca and compare with the CRAN msPCA reference."""
import sys
import time
import numpy as np
import pandas as pd
import pymspca
import binio

OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
cases = pd.read_csv("tests/fixtures/cases.csv")
rows = []
for cs in cases.itertuples():
    M = binio.read(f"tests/fixtures/{cs.file[:-4]}.bin")
    ks = [int(k) for k in str(cs.ks).split(";")]
    t0 = time.perf_counter()
    res = pymspca.mspca(M, cs.r, ks, type=cs.type, constraint=int(cs.ctype), max_iter=int(cs.maxIter),
                        max_restart_tpm=0, min_restart_tpm=0, scale=(str(cs.scale).upper() == "TRUE"), check_psd=False)
    py_t = time.perf_counter() - t0
    U_r = binio.read(f"{OUT}/{cs.id}_U.bin")
    m = binio.read(f"{OUT}/{cs.id}_meta.bin").ravel()
    meta = pd.Series(dict(objective=m[0], violation=m[1], total=m[2], runtime=m[3]))
    U = res.x_best
    # identical up to the sign of each component
    sgn = np.sign(np.sum(U * U_r, axis=0)); sgn[sgn == 0] = 1
    err = np.max(np.abs(U * sgn - U_r))
    supp = all(set(np.flatnonzero(U[:, j])) == set(np.flatnonzero(U_r[:, j])) for j in range(U.shape[1]))
    rows.append(dict(case=cs.id, data=cs.file.split(".")[0], type=cs.type, p=M.shape[1], r=cs.r, ks=cs.ks,
                     constraint=["orth", "uncorr"][int(cs.ctype)], max_abs_loading_diff=err, same_support=supp,
                     rel_obj_diff=abs(res.objective_value - meta.objective) / abs(meta.objective),
                     viol_py=res.feasibility_violation, viol_r=meta.violation, t_py=py_t, t_r=meta.runtime))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
exact = int((df.max_abs_loading_diff == 0).sum())
print(f"\nbit-identical: {exact}/{len(df)}; same support: {int(df.same_support.sum())}/{len(df)}; "
      f"worst loading diff {df.max_abs_loading_diff.max():.2e}")
