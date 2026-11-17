"""Sequential (R-identical) vs batched + multithreaded restarts: runtime and solution quality.

Run: python bench/restarts_speed.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

import pymspca

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from binio import read  # noqa: E402

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
S = read(ROOT / "tests/fixtures/S_snp500.bin")  # msPCA data-raw/snp500_corr, symmetrised
DATA = {
    "breast cancer, Sigma p=30": (np.corrcoef(load_breast_cancer().data, rowvar=False), "Sigma", 5, 5),
    "S&P 500, Sigma p=423": (S, "Sigma", 5, 10),
    "spiked, X n=80 p=2000": (read(ROOT / "tests/fixtures/X_spiked_n80_p2000.bin"), "X", 4, 15),
}
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
rows = []
for name, (M, typ, r, k) in DATA.items():
    for c, cname in ((0, "orthogonality"), (1, "uncorrelatedness")):
        for mode, threads in (("sequential", 1), ("batched", 1), ("batched", 0)):
            pymspca.set_num_threads(threads)
            for seed in SEEDS:
                t0 = time.perf_counter()
                res = pymspca.mspca(M, r, k, type=typ, constraint=c, random_state=seed, restarts=mode, check_psd=False)
                rows.append(dict(data=name, constraint=cname, mode=f"{mode} x{pymspca.get_num_threads()}",
                                 seed=seed, time_s=time.perf_counter() - t0, FVE=res.fraction_variance_explained,
                                 violation=res.feasibility_violation))
            print(rows[-1], flush=True)
df = pd.DataFrame(rows)
g = df.groupby(["data", "constraint", "mode"], sort=False).agg(time_s=("time_s", "mean"), FVE=("FVE", "mean"),
                                                               FVE_min=("FVE", "min"), viol_max=("violation", "max"))
base = g.xs("sequential x1", level="mode")["time_s"]
g["speedup"] = g["time_s"] / g.index.droplevel("mode").map(lambda i: base[i]).values
g["speedup"] = 1 / g["speedup"]
pd.set_option("display.width", 200)
print(g.to_string(float_format=lambda x: f"{x:.4g}"))
g.to_csv(ROOT / "bench/restarts_speed_results.csv")
