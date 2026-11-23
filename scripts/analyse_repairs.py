"""Analyse the step 6 repair runs (H3): accuracy against violation frontier, paired accuracy cost, target attainment.

Usage: python scripts/analyse_repairs.py            (reads results/repairs/runs/*.json, writes results/repairs/ and docs/repairs_results.md)

Definitions (fixed in docs/repairs.md before the main runs):
  violation rate   the largest of the three per-condition cell rates (monotone, convexity, calendar) on the DENSE grid of 500 test
                   surfaces (101 strikes x 20 maturities, every check is a stencil of three neighbouring points); the target of H3
                   is this rate below 1 percent. The base-grid surface rate on all 20,000 test surfaces is reported next to it.
  accuracy         implied vol RMSE and price RMSE on the 20,000 test surfaces (base grid).
  accuracy cost    paired difference to the squared-error control run with the same seed and noise level (same training set).
Intervals: 95 percent bootstrap over seeds (percentile), 4,000 resamples. With three to five seeds they are wide; read them as such.
"""
import csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data

RUN_DIR = os.path.join(data.ROOT, "results", "repairs", "runs")
OUT_DIR = os.path.join(data.ROOT, "results", "repairs")
DOC = os.path.join(data.ROOT, "docs", "repairs_results.md")
TARGET = 0.01
N_BOOT = 4000


def label(run):
    if run["deriv_weight"]:
        return ("slope loss" if run["deriv_terms"] == "slope" else "derivative loss"), run["deriv_weight"]
    if run["penalty_weight"]:
        return "arbitrage penalty", run["penalty_weight"]
    return "squared error", 0.0


def load(run_dir=RUN_DIR):
    rows = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        d = json.load(open(f))
        r, a = d["run"], d["results"]["all"]
        method, w = label(r)
        dense = d["dense_grid"]["violations"]
        rates = {k: dense[k]["cell_rate"] for k in ("monotone", "convexity", "calendar")}
        rows.append(dict(
            method=method, weight=w, noise=r["noise"], seed=r["seed"], tag=d["tag"], fit_seconds=d["fit_seconds"],
            rmse_iv=a["accuracy"]["rmse_iv"], p99_iv=a["accuracy"]["p99_abs_error_iv"], price_rmse=a["price_rmse"],
            unusable=a["violations"]["unusable_cell_share"],
            viol_dense=max(rates.values()), viol_monotone=rates["monotone"], viol_convexity=rates["convexity"], viol_calendar=rates["calendar"],
            surf_base=a["violations"]["any_violation_surface_rate"], surf_dense=dense["any_violation_surface_rate"],
            neg_mass=dense["negative_mass"]["mean_per_surface"],
            density_rmse_scaled=d["shape_error"]["density_rmse_scaled"], slope_rmse=d["shape_error"]["slope_rmse"],
            density_med_rel=d["shape_error"]["density_median_abs_rel_error"]))
    return rows


def boot(vals, rng):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    m = rng.choice(v, (N_BOOT, len(v))).mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def fmt(t, digits=4, pct=False):
    m, lo, hi = t
    f = (lambda x: f"{100 * x:.{max(digits - 2, 1)}f}") if pct else (lambda x: f"{x:.{digits}g}")
    return f"{f(m)} [{f(lo)}, {f(hi)}]"


def group(rows):
    g = {}
    for r in rows:
        g.setdefault((r["noise"], r["method"], r["weight"]), []).append(r)
    return g


def summary(rows, rng):
    g, out = group(rows), []
    for (noise, method, w), rs in sorted(g.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        ctrl = {r["seed"]: r for r in g.get((noise, "squared error", 0.0), [])}
        pairs = [(r, ctrl[r["seed"]]) for r in rs if r["seed"] in ctrl]
        row = dict(noise=noise, method=method, weight=w, n_seeds=len(rs))
        for k in ("rmse_iv", "price_rmse", "viol_dense", "viol_monotone", "viol_convexity", "viol_calendar", "surf_base", "surf_dense",
                  "neg_mass", "density_rmse_scaled", "slope_rmse", "density_med_rel", "unusable", "fit_seconds"):
            row[k] = boot([r[k] for r in rs], rng)
        if method != "squared error" and pairs:
            for k in ("rmse_iv", "price_rmse"):
                row[k + "_cost"] = boot([a[k] - b[k] for a, b in pairs], rng)
        out.append(row)
    return out


def write_csv(path, summ):
    cols = [k for k in summ[0] if isinstance(summ[0][k], (str, int, float))]
    stat = [k for k in summ[0] if isinstance(summ[0][k], tuple)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + [f"{k}_{s}" for k in stat for s in ("mean", "lo", "hi")])
        for r in summ:
            w.writerow([r[c] for c in cols] + [x for k in stat for x in r.get(k, (float("nan"),) * 3)])


def table(summ, noise):
    head = ("| method | weight | seeds | violating cells, dense (%) | surfaces violating, base (%) | IV RMSE (vol pts) | price RMSE (1e-3) | "
            "density error (scaled) | IV cost (vol pts) |\n|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in summ:
        if r["noise"] != noise:
            continue
        cost = fmt(tuple(100 * x for x in r["rmse_iv_cost"]), 3) if "rmse_iv_cost" in r else "-"
        lines.append(f"| {r['method']} | {r['weight']:g} | {r['n_seeds']} | {fmt(tuple(100 * x for x in r['viol_dense']), 3)} | "
                     f"{fmt(tuple(100 * x for x in r['surf_base']), 3)} | {fmt(tuple(100 * x for x in r['rmse_iv']), 3)} | "
                     f"{fmt(tuple(1e3 * x for x in r['price_rmse']), 3)} | {fmt(r['density_rmse_scaled'], 3)} | {cost} |")
    return head + "\n".join(lines) + "\n"


def target_attainment(summ, rows, rng):
    """For each noise level and repair family: the weights whose mean dense-grid violation rate is below the target, the cheapest of
    them by mean paired IV cost, and the paired difference of costs between the derivative loss and the penalty."""
    out = []
    for noise in sorted({r["noise"] for r in summ}):
        best = {}
        for fam in ("derivative loss", "arbitrage penalty"):
            ok = [r for r in summ if r["noise"] == noise and r["method"] == fam and r["viol_dense"][0] < TARGET]
            best[fam] = min(ok, key=lambda r: r["rmse_iv_cost"][0]) if ok else None
        line = dict(noise=noise, **{fam: (None if b is None else (b["weight"], b["rmse_iv_cost"], b["price_rmse_cost"], b["viol_dense"]))
                                    for fam, b in best.items()})
        if all(best.values()):
            g = group(rows)
            seeds = sorted({r["seed"] for r in g[(noise, "squared error", 0.0)]})
            diffs = {}
            for k in ("rmse_iv", "price_rmse"):
                dv = {r["seed"]: r[k] for r in g[(noise, "derivative loss", best["derivative loss"]["weight"])]}
                pn = {r["seed"]: r[k] for r in g[(noise, "arbitrage penalty", best["arbitrage penalty"]["weight"])]}
                common = [s for s in seeds if s in dv and s in pn]
                diffs[k] = boot([dv[s] - pn[s] for s in common], rng) if common else None
            line["difference_derivative_minus_penalty"] = diffs
        out.append(line)
    return out


def figure(summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noises = sorted({r["noise"] for r in summ})
    fig, axes = plt.subplots(2, len(noises), figsize=(5.2 * len(noises), 7.4), squeeze=False)
    style = {"squared error": ("k", "o"), "derivative loss": ("tab:blue", "s"), "slope loss": ("tab:cyan", "D"), "arbitrage penalty": ("tab:red", "^")}
    for j, noise in enumerate(noises):
        for i, (key, ylab) in enumerate((("rmse_iv", "implied vol RMSE (vol points)"), ("price_rmse", "price RMSE (1e-3)"))):
            ax = axes[i, j]
            for method in style:
                rs = sorted([r for r in summ if r["noise"] == noise and r["method"] == method], key=lambda r: r["weight"])
                if not rs:
                    continue
                x = [max(r["viol_dense"][0], 1e-6) * 100 for r in rs]
                y = [r[key][0] * (100 if key == "rmse_iv" else 1e3) for r in rs]
                lo = [r[key][1] * (100 if key == "rmse_iv" else 1e3) for r in rs]
                hi = [r[key][2] * (100 if key == "rmse_iv" else 1e3) for r in rs]
                ax.errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - y], color=style[method][0], marker=style[method][1], label=method,
                            capsize=2, linestyle="-" if len(rs) > 1 else "none")
                for r, xi, yi in zip(rs, x, y):
                    if r["weight"]:
                        ax.annotate(f"{r['weight']:g}", (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax.axvline(TARGET * 100, color="grey", linestyle=":")
            ax.set_xscale("log")
            ax.set_xlabel("violating cells, dense grid (%, floor 1e-4)")
            ax.set_ylabel(ylab)
            ax.set_title(f"label noise {noise * 100:g}%")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    rows = load()
    if not rows:
        sys.exit(f"no runs in {RUN_DIR}")
    rng = np.random.default_rng(0)
    summ = summary(rows, rng)
    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUT_DIR, "summary.csv"), summ)
    figure(summ, os.path.join(OUT_DIR, "frontier.png"))
    att = target_attainment(summ, rows, rng)
    json.dump(att, open(os.path.join(OUT_DIR, "target_attainment.json"), "w"), indent=2, default=float)
    parts = ["# H3 results: derivative loss against arbitrage penalty (generated by scripts/analyse_repairs.py)\n",
             f"{len(rows)} runs. Violation target: dense-grid violating-cell rate below {TARGET * 100:g} percent. Values are means with 95 percent "
             "bootstrap intervals over seeds; the cost is paired with the squared-error control of the same seed. Read the intervals: few seeds.\n"]
    for noise in sorted({r["noise"] for r in summ}):
        parts.append(f"\n## Label noise {noise * 100:g} percent\n\n" + table(summ, noise))
    parts.append("\n## Weights reaching the target, and the cheapest of them\n")
    for a in att:
        parts.append(f"\nNoise {a['noise'] * 100:g} percent:\n")
        for fam in ("derivative loss", "arbitrage penalty"):
            b = a[fam]
            parts.append(f"- {fam}: " + ("no weight reached the target" if b is None else
                         f"weight {b[0]:g}, IV cost {fmt(tuple(100 * x for x in b[1]), 3)} vol points, price cost {fmt(tuple(1e3 * x for x in b[2]), 3)} (1e-3), "
                         f"violating cells {fmt(tuple(100 * x for x in b[3]), 3)} percent"))
        d = a.get("difference_derivative_minus_penalty")
        if d:
            parts.append(f"- derivative minus penalty, paired by seed: IV RMSE {fmt(tuple(100 * x for x in d['rmse_iv']), 3)} vol points, "
                         f"price RMSE {fmt(tuple(1e3 * x for x in d['price_rmse']), 3)} (1e-3); negative means the derivative loss is more accurate")
    parts.append("\n![frontier](../results/repairs/frontier.png)\n")
    open(DOC, "w").write("\n".join(parts))
    print(f"wrote {DOC}, {OUT_DIR}/summary.csv, frontier.png")


if __name__ == "__main__":
    main()
