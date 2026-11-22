"""Compare the scalar reference pricer (src/heston_reference.py) with the vectorised pricer (src/heston.py).

Draws parameter sets from the shipped data and prints the median absolute call-price difference by maturity (rows)
and strike (columns). See the last section of docs/heston_pricer.md for how to read the result.
"""
import argparse, gzip, os, sys, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import heston as H
import heston_reference as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/HestonTrainSet.txt.gz")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    P = np.load(gzip.GzipFile(a.data, "r"))[:, :5]
    idx = np.random.default_rng(a.seed).choice(len(P), a.n, replace=False)
    ts, ks = (0.1, 0.6, 2.0), (0.5, 1.0, 1.5)
    fast = H.call_prices(P[idx], ks, ts)
    d = np.array([[[R.heston_call(1.0, k, t, *P[i]) - fast[n, a_, b]
                    for b, k in enumerate(ks)] for a_, t in enumerate(ts)] for n, i in enumerate(idx)])
    np.set_printoptions(linewidth=150, precision=1)
    print(f"median |reference - vectorised| call price over {a.n} shipped parameter sets")
    print("rows: maturity", ts, " columns: strike", ks)
    print(np.median(abs(d), axis=0))
    print("largest difference:", float(abs(d).max()))


if __name__ == "__main__":
    main()
