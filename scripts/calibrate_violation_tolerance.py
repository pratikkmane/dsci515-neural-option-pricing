"""Calibrate the violation tolerance on exact Heston surfaces (step 3).

Rule: tolerance = 10 x the largest violation margin seen on exact surfaces in a calibration sample, rounded up to a power
of ten. It is then checked on fresh exact surfaces (different seeds, plus a near-the-bounds subset like the planned test
set), where the observed false-positive rate should be zero.

Also reports (a) how often an injected, known-size dent would be detected and (b) how much headroom the true surfaces have
before a violation, which shows where tiny errors can flip the sign.

Writes results/violation_tolerance.json. Needs only numpy and scipy.
"""
import argparse, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import heston as H
import violations as V

SAFETY = 10.0


def boundary_params(n, seed, band=0.10):
    """Parameter sets with at least one parameter within `band` of its range of a bound (like the planned boundary subset)."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, 5))
    j = rng.integers(0, 5, n)
    edge = rng.integers(0, 2, n)
    u[np.arange(n), j] = np.where(edge == 0, band * rng.random(n), 1 - band * rng.random(n))
    return H.LB + u * (H.UB - H.LB)


def surfaces(params):
    """The three ways a surface reaches the detector: dense prices, base-grid prices, base grid via implied vol."""
    dense = H.call_prices(params, H.DENSE_STRIKES, H.DENSE_MATURITIES)
    base = H.call_prices(params)
    via_iv = V.iv_to_call(H.implied_vols(params), H.STRIKES, H.MATURITIES)
    return {
        "dense_direct": (dense, H.DENSE_STRIKES, H.DENSE_MATURITIES),
        "base_direct": (base, H.STRIKES, H.MATURITIES),
        "base_via_implied_vol": (via_iv, H.STRIKES, H.MATURITIES),
    }


def max_margins(sets):
    out = {}
    for name, (c, k, t) in sets.items():
        m = V.margins(c, k, t)
        out[name] = {cond: float(np.nanmax(m[cond])) for cond in V.CONDITIONS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-boundary", type=int, default=1000)
    ap.add_argument("--out", default="results/violation_tolerance.json")
    a = ap.parse_args()
    t0 = time.time()
    res = {"rule": f"tolerance = {SAFETY:g} x largest exact-surface margin in the calibration sample, rounded up to a power of ten"}

    # ---- calibration sample: 32 corners of the parameter box + Sobol sets
    corners = np.array(list(itertools.product(*[(H.LB[i], H.UB[i]) for i in range(5)])))
    cal = np.vstack([corners, H.sobol_params(a.n_cal, seed=101)])
    cal_max = max_margins(surfaces(cal))
    pooled = max(v for d in cal_max.values() for v in d.values())
    tol = 10.0 ** np.ceil(np.log10(SAFETY * pooled))
    res["calibration"] = dict(n_param_sets=len(cal), includes_32_corners=True, max_margin_by_source=cal_max,
                              largest_margin=pooled, tolerance=float(tol))

    # ---- validation on fresh exact surfaces
    rng = np.random.default_rng(202)
    fresh = {
        "sobol_seed_202": H.sobol_params(a.n_val, seed=202),
        "uniform_random": H.LB + rng.random((a.n_val, 5)) * (H.UB - H.LB),
        "near_bounds": boundary_params(a.n_boundary, seed=303),
    }
    val = {}
    for label, params in fresh.items():
        sets = surfaces(params)
        entry = {}
        for name, (c, k, t) in sets.items():
            m = V.margins(c, k, t)
            checks = int(sum(np.sum(~np.isnan(m[cond])) for cond in V.CONDITIONS))
            flagged = int(sum(np.sum(m[cond] > tol) for cond in V.CONDITIONS))
            entry[name] = dict(checks=checks, false_positives=flagged,
                               largest_margin=float(max(np.nanmax(m[cond]) for cond in V.CONDITIONS)),
                               false_positive_rate_95pct_upper_bound=float(3.0 / checks))
        val[label] = entry
    res["validation_at_tolerance"] = val
    res["validation_total_false_positives"] = int(sum(e[s]["false_positives"] for e in val.values() for s in e))

    # ---- sensitivity: inject a dent of known size at one interior point and see if the convexity check flags it
    k, t = H.DENSE_STRIKES, H.DENSE_MATURITIES
    base_c = H.call_prices(H.sobol_params(300, seed=404), k, t)
    rng = np.random.default_rng(5)
    rows = {}
    for delta in (1e-12, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3):
        hit = 0
        total = 0
        for n in range(len(base_c)):
            for _ in range(20):
                j, i = rng.integers(0, len(t)), rng.integers(1, len(k) - 1)
                c = base_c[n:n + 1].copy()
                c[0, j, i] += delta
                hit += bool(V.flags(c, k, t, tol)["convexity"][0, j, i - 1])
                total += 1
        rows[f"{delta:.0e}"] = hit / total
    res["injected_dent_detection_rate_dense_grid"] = rows

    # ---- headroom: price of the true butterfly (margin to a violation) on exact surfaces
    sets = surfaces(H.sobol_params(500, seed=505))
    head = {}
    for name in ("base_direct", "dense_direct"):
        c, kk, tt = sets[name]
        b = -V.margins(c, kk, tt)["convexity"]                 # butterfly price, positive on a valid surface
        bands = {"1e-12": 1e-12, "1e-9": 1e-9, "1e-6": 1e-6, "1e-4": 1e-4}
        head[name] = {f"share_of_stencils_with_butterfly_price_below_{lab}": float(np.mean(b < v)) for lab, v in bands.items()}
        head[name]["by_maturity_share_below_1e-6"] = [round(float(np.mean(b[:, j, :] < 1e-6)), 4) for j in range(b.shape[1])]
        head[name]["maturities"] = [float(x) for x in tt]
    res["headroom_exact_surfaces"] = head
    res["seconds"] = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("largest exact-surface margin (calibration):", f"{pooled:.2e}", "-> tolerance", f"{tol:.0e}")
    print("false positives on fresh exact surfaces:", res["validation_total_false_positives"])
    for label, e in val.items():
        for s, d in e.items():
            print(f"  {label:15s} {s:22s} checks {d['checks']:>10,}  false positives {d['false_positives']}  largest margin {d['largest_margin']:.1e}")
    print("injected dent detection rate:", rows)
    print("seconds", round(res["seconds"], 1))


if __name__ == "__main__":
    main()
