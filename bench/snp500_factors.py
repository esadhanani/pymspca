"""Sparse statistical factors in S&P 500 returns (msPCA case-study data).

Input: msPCA's `snp500` dataset, the market-deflated correlation matrix of daily log-returns
for 423 constituents, 2010-2019. With 10 names per factor, each sparse PC should be a
recognisable industry basket. We quantify that with GICS sector purity (share of a factor's
names in its modal sector; sectors from the current S&P 500 constituent list, 403/423 mapped)
and compare against the same-size baskets formed from the top-|loading| names of dense PCA.

Run: python bench/snp500_factors.py
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import pymspca

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from binio import read  # noqa: E402

warnings.simplefilter("ignore", pymspca.MSPCAWarning)
S = read(ROOT / "tests/fixtures/S_snp500.bin")
tickers = np.array([l.strip() for l in open(ROOT / "bench/snp500_tickers.txt")])
sector = dict(pd.read_csv(ROOT / "bench/snp500_sectors.csv").values)
R, K = 8, 10


def purity(idx):
    secs = [sector[t] for t in tickers[idx] if t in sector]
    top, cnt = Counter(secs).most_common(1)[0]
    return top, cnt / len(secs)


res = pymspca.mspca(S, R, K, constraint="orthogonality", restarts="batched", random_state=0)
ev, V = np.linalg.eigh(S)
V = V[:, ::-1]
print(f"msPCA, {R} factors x {K} names: FVE {res.fraction_variance_explained:.3f} "
      f"(dense PCA with all 423 names: {ev[::-1][:R].sum() / np.trace(S):.3f}), "
      f"orthogonality violation {res.feasibility_violation:.1e}\n")
rows = []
for j in range(R):
    u = res.x_best[:, j]
    idx = np.flatnonzero(u)
    idx = idx[np.argsort(-np.abs(u[idx]))]
    sec, pur = purity(idx)
    didx = np.argsort(-np.abs(V[:, j]))[:K]
    dsec, dpur = purity(didx)
    rows.append(dict(factor=f"PC{j + 1}", sector=sec, purity=pur, dense_top10_sector=dsec, dense_top10_purity=dpur,
                     names=" ".join(tickers[idx])))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(df.to_string(index=False, float_format=lambda x: f"{x:.0%}"))
print(f"\nmean sector purity: msPCA {df.purity.mean():.0%} vs dense-PCA top-{K} baskets {df.dense_top10_purity.mean():.0%}")
df.to_csv(ROOT / "bench/snp500_factors_results.csv", index=False)
