import glob, pandas as pd, binio
for f in glob.glob("parity/data/[SX]_*.csv"):
    binio.write("tests/fixtures/" + f.split("/")[-1][:-4] + ".bin", pd.read_csv(f, float_precision="round_trip").to_numpy(float))
