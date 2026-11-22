"""Validate src/heston.py.

1. Pricer against an independent reference: the textbook (Albrecher) characteristic function, integrated with
   scipy's adaptive quad, on Sobol parameter sets and on the 32 corners of the parameter box.
2. Pricer against the implied vols shipped in data/HestonTrainSet.txt.gz, per maturity and per strike.
3. How many grid cells are numerically unresolved (out-of-the-money price below heston.PRICE_FLOOR).

Writes results/heston_validation.json and results/heston_vs_shipped.png.
"""
import argparse, gzip, itertools, json, os, sys, time
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import heston as H


def ref_otm_price(p, t, k):
    """Out-of-the-money price by quad on the textbook formula (no cancellation-avoiding rewrites)."""
    xi0, rho, sig, th, kap = p

    def phi(z):
        a = kap - 1j * rho * sig * z
        d = np.sqrt(a * a + sig ** 2 * (z * z + 1j * z))
        g = (a - d) / (a + d)
        e = np.exp(-d * t)
        return np.exp(kap * th / sig ** 2 * ((a - d) * t - 2 * np.log((1 - g * e) / (1 - g)))
                      + xi0 / sig ** 2 * (a - d) * (1 - e) / (1 - g * e))

    f = lambda u: (np.exp(1j * u * np.log(k)) * phi(u - 0.5j)).real / (u * u + 0.25)
    # the integrand can decay very slowly at the corners, so integrate piecewise out to a large cutoff
    x = 0.0
    edges = [0, 1, 5, 20, 100, 500, 2000, 10000, 50000, 200000]
    for a, b in zip(edges[:-1], edges[1:]):
        x += quad(f, a, b, limit=2000, epsabs=1e-15, epsrel=1e-13)[0]
    x *= np.sqrt(k) / np.pi
    return 1 - x if k >= 1 else k - x


def ref_otm_price_mp(p, t, k, dps=30):
    """Same textbook formula in 30-digit arithmetic (mpmath), the tie-breaker when double-precision quad disagrees."""
    import mpmath as mp
    mp.mp.dps = dps
    xi0, rho, sig, th, kap = [mp.mpf(float(x)) for x in p]

    def phi(z):
        a = kap - 1j * rho * sig * z
        d = mp.sqrt(a * a + sig ** 2 * (z * z + 1j * z))
        g = (a - d) / (a + d)
        e = mp.exp(-d * t)
        return mp.exp(kap * th / sig ** 2 * ((a - d) * t - 2 * mp.log((1 - g * e) / (1 - g)))
                      + xi0 / sig ** 2 * (a - d) * (1 - e) / (1 - g * e))

    f = lambda u: (mp.exp(1j * u * mp.log(k)) * phi(u - 0.5j)).real / (u * u + 0.25)
    x = float(mp.sqrt(k) / mp.pi * mp.quad(f, [0, 1, 5, 20, 100, 500, 2000, 10000, 50000, 200000, mp.inf]))
    return 1 - x if k >= 1 else k - x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/HestonTrainSet.txt.gz")
    ap.add_argument("--out", default="results/heston_validation.json")
    ap.add_argument("--n-random", type=int, default=24)
    a = ap.parse_args()
    res = {}

    # 1. independent reference
    corners = np.array(list(itertools.product(*[(H.LB[i], H.UB[i]) for i in range(5)])))
    pts = np.vstack([corners, H.sobol_params(a.n_random, seed=11)])
    ts, ks = (0.1, 0.6, 2.0), (0.5, 1.0, 1.5)
    vec = H.otm_prices(pts, ks, ts)
    errs = np.array([[[abs(vec[i, j, l] - ref_otm_price(p, t, k)) for l, k in enumerate(ks)]
                      for j, t in enumerate(ts)] for i, p in enumerate(pts)])
    res["reference_check"] = dict(n_param_sets=len(pts), n_prices=int(errs.size), max_abs_price_error=float(errs.max()),
                                  median_abs_price_error=float(np.median(errs)),
                                  reference="textbook characteristic function with scipy quad, piecewise to u=2e5")
    print("1. vs reference:", res["reference_check"])
    # The double-precision reference itself loses digits where sigma is small, so recheck the worst cases at 30 digits
    try:
        worst = np.dstack(np.unravel_index(np.argsort(-errs, axis=None)[:6], errs.shape))[0]
        e_vec, e_ref = [], []
        for i, j, l in worst:
            m = ref_otm_price_mp(pts[i], ts[j], ks[l])
            e_vec.append(abs(vec[i, j, l] - m)); e_ref.append(abs(ref_otm_price(pts[i], ts[j], ks[l]) - m))
        res["worst_cases_at_30_digits"] = dict(n_cases=len(worst), max_abs_error_of_heston_py=float(max(e_vec)),
                                               max_abs_error_of_double_precision_reference=float(max(e_ref)))
        print("   worst", len(worst), "cases at 30 digits:", res["worst_cases_at_30_digits"])
    except ImportError:
        print("   mpmath not installed: skipping the 30-digit check (pip install mpmath)")

    # 2. against the shipped data
    dat = np.load(gzip.GzipFile(a.data, "r"))
    P, IV = dat[:, :5], dat[:, 5:].reshape(-1, len(H.MATURITIES), len(H.STRIKES))
    t0 = time.time()
    price = H.otm_prices(P)
    ours = H.implied_vol_otm(price, H.STRIKES[None, None, :], H.MATURITIES[None, :, None])
    res["seconds_for_shipped_set"] = time.time() - t0
    d = IV - ours
    res["unresolved_cell_share_percent"] = (100 * (price < H.PRICE_FLOOR).mean(0)).round(2).tolist()
    res["unresolved_cells_total"] = int(np.isnan(ours).sum())
    res["cells_total"] = int(ours.size)
    res["shipped_minus_exact_by_maturity"] = {
        str(t): dict(mean=float(np.nanmean(d[:, j])), rms=float(np.sqrt(np.nanmean(d[:, j] ** 2))),
                     p99_abs=float(np.nanquantile(abs(d[:, j]), 0.99)), max_abs=float(np.nanmax(abs(d[:, j]))))
        for j, t in enumerate(H.MATURITIES)}
    res["shipped_minus_exact_rms_by_cell"] = np.sqrt(np.nanmean(d ** 2, axis=0)).round(6).tolist()
    res["negative_otm_prices"] = dict(count=int((price < 0).sum()), min=float(price.min()))
    print("2. shipped minus exact, by maturity:")
    for t, v in res["shipped_minus_exact_by_maturity"].items():
        print(f"   T={t}: mean {v['mean']:+.2e}  rms {v['rms']:.2e}  p99 {v['p99_abs']:.2e}  max {v['max_abs']:.2e}")
    print(f"3. unresolved cells: {res['unresolved_cells_total']} of {res['cells_total']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    im = ax[0].imshow(np.sqrt(np.nanmean(d ** 2, axis=0)) * 100); plt.colorbar(im, ax=ax[0], format="%.2f")
    ax[0].set_title("RMS(shipped - exact), volatility points")
    im = ax[1].imshow(100 * (price < H.PRICE_FLOOR).mean(0)); plt.colorbar(im, ax=ax[1], format="%.0f")
    ax[1].set_title("Unresolved cells, percent (OTM price < %.0e)" % H.PRICE_FLOOR)
    for axi in ax:
        axi.set_xticks(range(len(H.STRIKES))); axi.set_xticklabels(H.STRIKES, fontsize=7)
        axi.set_yticks(range(len(H.MATURITIES))); axi.set_yticklabels(H.MATURITIES, fontsize=7)
        axi.set_xlabel("Strike"); axi.set_ylabel("Maturity")
    plt.tight_layout(); plt.savefig(a.out.replace(".json", "").replace("heston_validation", "heston_vs_shipped") + ".png", dpi=150)


if __name__ == "__main__":
    main()
