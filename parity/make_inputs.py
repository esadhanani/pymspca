"""Build parity inputs + case table shared by the R and Python runners."""
import csv
import numpy as np
import pandas as pd

D = "parity/data"
rng = np.random.default_rng(20261115)

# S&P 500 market-deflated correlation (msPCA data-raw), symmetrised exactly as data-raw/snp500.R
S = pd.read_csv(f"{D}/snp500_corr.csv", index_col=0).to_numpy(float)
S = (S + S.T) / 2
pd.DataFrame(S).to_csv(f"{D}/S_snp500.csv", index=False)

# Spiked-covariance data, n << p (the raw-data path's target regime)
def spiked(n, p, r, k, snr):
    U = np.zeros((p, r))
    for j in range(r):
        U[j * k:(j + 1) * k, j] = rng.choice([-1, 1], k) / np.sqrt(k)
    Z = rng.standard_normal((n, r)) * np.sqrt(snr)
    return Z @ U.T + rng.standard_normal((n, p))

pd.DataFrame(spiked(120, 400, 3, 10, 6.0)).to_csv(f"{D}/X_spiked_n120_p400.csv", index=False)
pd.DataFrame(spiked(500, 60, 3, 8, 3.0)).to_csv(f"{D}/X_spiked_n500_p60.csv", index=False)
Xs = spiked(80, 2000, 4, 15, 8.0)
pd.DataFrame(Xs).to_csv(f"{D}/X_spiked_n80_p2000.csv", index=False)

cases = []
def add(file, typ, r, ks, ctype, scale=True, max_iter=200):
    cases.append(dict(id=f"c{len(cases):02d}", file=file, type=typ, r=r, ks=";".join(map(str, ks)),
                      ctype=ctype, scale=str(scale).upper(), maxIter=max_iter))

for c in (0, 1):
    add("S_mtcarscor.csv", "Sigma", 2, [4, 4], c)
    add("S_mtcarscor.csv", "Sigma", 3, [5, 3, 2], c)
    add("X_mtcars.csv", "X", 3, [4, 4, 4], c)
    add("X_mtcars.csv", "X", 3, [4, 4, 4], c, scale=False)
    add("S_snp500.csv", "Sigma", 5, [10] * 5, c)
    add("S_snp500.csv", "Sigma", 10, [20] * 10, c)
    add("X_spiked_n120_p400.csv", "X", 3, [10, 10, 10], c)
    add("X_spiked_n500_p60.csv", "X", 4, [8, 8, 8, 5], c)
    add("X_spiked_n80_p2000.csv", "X", 4, [15] * 4, c)

with open("parity/cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cases[0]))
    w.writeheader()
    w.writerows(cases)
print(len(cases), "cases")
