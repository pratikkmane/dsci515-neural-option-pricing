"""Aggregate the experiment runs into tables, confidence intervals, the H1 and H2 analyses, bias and variance, and figures (step 5).

Reads results/runs/*.json and *.npz. Writes
  results/tables/settings.csv       one row per setting (model, configuration, size, noise) with 95 percent intervals
  results/tables/h1_auc.csv, h2_trends.csv, bias_variance.csv
  results/figures/*.png
  docs/experiments_results.md       generated tables (do not edit by hand; the protocol and interpretation are in docs/experiments.md)

Intervals are percentile bootstrap intervals, 95 percent. Seeds are independent training sets (a different Sobol sample each),
so resampling seeds captures training-set variation. Rates over test surfaces use a two-stage bootstrap: seeds and test surfaces
are resampled together. Differences between settings are paired by seed and by surface. Nothing here corrects for the many
comparisons shown: read the intervals as descriptive, and look for effects that are large and consistent, not for isolated hits.
"""
import argparse, csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data

RNG_SEED = 12345
B_RATE = 1000          # bootstrap replicates for rates and differences
B_BV = 300             # for the bias-variance correlation
FPR_PER_SURFACE = 2.2e-5     # 95 percent upper bound: 9.5e-8 per check (docs/violations.md) x 229 stencils of a base-grid surface
KEY_FIELDS = ("model", "target", "activation", "width", "depth", "epochs", "schedule", "n", "noise")


def config_key(run):
    return tuple(run[k] for k in KEY_FIELDS)


def load_runs(run_dir):
    groups = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        with open(path) as f:
            s = json.load(f)
        groups.setdefault(config_key(s["run"]), []).append(s)
    for v in groups.values():
        v.sort(key=lambda s: s["run"]["seed"])
    return groups


def name_of(key):
    model, target, act, width, depth, epochs, sched, n, noise = key
    if model == "grid_net":
        return f"grid_net {act} w{width}d{depth}"
    if model == "pointwise":
        return f"pointwise {target} {act} w{width}d{depth}"
    return model


def pct(x):
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.1f}%"


REGION_N = {}          # surfaces per region of the test set, filled in main(): "all", "interior", "boundary"


def ci_str(t, scale=1.0, digits=4, percent=False, region=None, seeds=None):
    """Format (mean, lo, hi). For a rate that is exactly zero a bootstrap interval is [0, 0] and means nothing, so when the region
    and number of seeds are given it shows the rule-of-three 95 percent upper bound on the rate (surfaces treated as independent)."""
    m, lo, hi = t
    if np.isnan(m):
        return "n/a"
    if percent and m == 0 and region and seeds:
        return f"0 (<{100 * 3 / (seeds * REGION_N[region]):.3f}%)"
    if percent:
        return f"{100 * m:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{m * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


# ---------------------------------------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------------------------------------

def seed_boot(values, rng, n_boot=2000):
    """Mean and 95 percent percentile interval over seeds (NaN values dropped)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    if len(v) == 1:
        return (float(v[0]),) * 3
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_rates(flag_sets, masks, rng, n_boot=B_RATE):
    """Resampled mean violation rates for several settings with common resampling of seeds and test surfaces.

    flag_sets: list of (S, N) bool arrays (same S seeds, same N surfaces). masks: list of boolean (N,) selecting surfaces for each.
    Returns (K, n_boot): rates under each bootstrap replicate; point estimates are the plain means.
    """
    s, n = flag_sets[0].shape
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float32)        # (B, N)
    seed_idx = rng.integers(0, s, (n_boot, s))
    out = []
    for f, m in zip(flag_sets, masks):
        w = counts * m[None, :]
        denom = w.sum(axis=1)                                                                # surfaces in the resample
        per_seed = (f.astype(np.float32) @ w.T) / denom[None, :]                             # (S, B)
        out.append(per_seed[seed_idx, np.arange(n_boot)[:, None]].mean(axis=1))
    return np.array(out)


def interval(x):
    return float(np.mean(x)), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


# ---------------------------------------------------------------------------------------------------------------------
# Bias and variance
# ---------------------------------------------------------------------------------------------------------------------

def bias_variance(preds, truth, sel=None):
    """Decompose mean squared error across seeds (independent training sets) into squared bias and variance.

    preds (S, M, 88) predicted implied vols, truth (M, 88). Only cells that are valid for every seed and resolved in the truth
    count. MSE (averaged over seeds and cells) = bias^2 + variance exactly, with variance the population variance across seeds."""
    ok = ~np.isnan(truth) & ~np.isnan(preds).any(axis=0)
    if sel is not None:
        ok &= sel[:, None]
    if not ok.any() or preds.shape[0] < 2:
        return dict(bias2=float("nan"), variance=float("nan"), mse=float("nan"), cells=0, bias2_corrected=float("nan"), variance_unbiased=float("nan"))
    p = preds[:, ok].astype(np.float64)
    t = truth[ok]
    mean = p.mean(axis=0)
    bias2 = float(np.mean((mean - t) ** 2))
    var = float(np.mean(p.var(axis=0)))
    mse = float(np.mean((p - t[None]) ** 2))
    # The plug-in bias^2 of the seed average is inflated by (unbiased variance) / S, and the population variance is too small by
    # a factor (S - 1) / S. The identity above is exact for the plug-in values; the corrected values estimate the true terms.
    s_n = preds.shape[0]
    var_unbiased = var * s_n / (s_n - 1)
    return dict(bias2=bias2, variance=var, mse=mse, cells=int(ok.sum()),
                bias2_corrected=bias2 - var_unbiased / s_n, variance_unbiased=var_unbiased)


# ---------------------------------------------------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------------------------------------------------

def aggregate(groups, test_region, truth_sub, run_dir, rng):
    interior = test_region == "interior"
    rows, store = [], {}
    for key, runs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1:7]), kv[0][7], kv[0][8])):
        S = len(runs)
        tags = [r["tag"] for r in runs]
        npz = [np.load(os.path.join(run_dir, t + ".npz")) for t in tags]
        flags = np.stack([z["viol"] for z in npz])                                           # (S, N)
        preds = np.stack([z["pred_iv"] for z in npz]).astype(np.float64)                     # (S, M, 88)
        r_all, r_in, r_out = boot_rates([flags] * 3, [np.ones(flags.shape[1], bool), interior, ~interior], rng)
        acc = lambda f: [f(r) for r in runs]
        row = dict(
            name=name_of(key), model=key[0], target=key[1], activation=key[2], width=key[3], depth=key[4], epochs=key[5],
            schedule=key[6], n=key[7], noise=key[8], seeds=S,
            rmse_iv=seed_boot(acc(lambda r: r["results"]["all"]["accuracy"]["rmse_iv"]), rng),
            p99_abs_err=seed_boot(acc(lambda r: r["results"]["all"]["accuracy"]["p99_abs_error_iv"]), rng),
            viol_all=interval(r_all), viol_interior=interval(r_in), viol_boundary=interval(r_out),
            conv_rate=seed_boot(acc(lambda r: r["results"]["all"]["violations"]["convexity"]["surface_rate"]), rng),
            cal_rate=seed_boot(acc(lambda r: r["results"]["all"]["violations"]["calendar"]["surface_rate"]), rng),
            mono_rate=seed_boot(acc(lambda r: r["results"]["all"]["violations"]["monotone"]["surface_rate"]), rng),
            sev_1e6=seed_boot(acc(lambda r: r["results"]["all"]["violations"]["severity_surface_rate"]["1e-06"]), rng),
            sev_1e4=seed_boot(acc(lambda r: r["results"]["all"]["violations"]["severity_surface_rate"]["1e-04"]), rng),
            unusable=seed_boot(acc(lambda r: r["results"]["all"]["violations"]["unusable_cell_share"]), rng),
            price_rmse=seed_boot(acc(lambda r: r["results"]["all"].get("price_rmse", float("nan"))), rng),
            neg_mass=seed_boot(acc(lambda r: r["results"]["all"]["violations"]["negative_mass"]["mean_per_surface"]), rng),
            fit_seconds=float(np.mean(acc(lambda r: r["fit_seconds"]))),
        )
        for m in ("pointwise_price", "pointwise_iv", "curvature_abs", "curvature_rel"):
            row["auc_" + m] = seed_boot(acc(lambda r: r["local_error_auc"]["auc"][m]), rng)
        d_primary = [r["local_error_auc"]["auc"]["curvature_rel"] - r["local_error_auc"]["auc"]["pointwise_iv"] for r in runs]
        d_secondary = [r["local_error_auc"]["auc"]["curvature_abs"] - r["local_error_auc"]["auc"]["pointwise_price"] for r in runs]
        row["auc_diff_primary"] = seed_boot(d_primary, rng)
        row["auc_diff_secondary"] = seed_boot(d_secondary, rng)
        row["auc_positives_mean"] = float(np.mean([r["local_error_auc"]["n_positive"] for r in runs]))
        bv_all = bias_variance(preds, truth_sub, None)
        bv_in = bias_variance(preds, truth_sub, interior[:truth_sub.shape[0]])
        bv_out = bias_variance(preds, truth_sub, ~interior[:truth_sub.shape[0]])
        row.update(bias2=bv_all["bias2"], variance=bv_all["variance"], bv_mse=bv_all["mse"], bias2_corrected=bv_all["bias2_corrected"],
                   variance_unbiased=bv_all["variance_unbiased"],
                   var_interior=bv_in["variance"], var_boundary=bv_out["variance"], bias2_interior=bv_in["bias2"], bias2_boundary=bv_out["bias2"])
        if key[0] == "pointwise":
            dg = [r["dense_grid"] for r in runs]
            row["dense_viol"] = seed_boot([d["violations"]["any_violation_surface_rate"] for d in dg], rng)
            row["neg_density_share"] = seed_boot([d["autodiff_curvature"]["negative_density_share"] for d in dg], rng)
        rows.append(row)
        store[key] = dict(flags=flags, preds=preds, runs=runs)
    return rows, store


def flat(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, tuple):
            out[k] = v[0]; out[k + "_lo"] = v[1]; out[k + "_hi"] = v[2]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------------------------------------------------
# H2 trends and the variance link
# ---------------------------------------------------------------------------------------------------------------------

def h2_trends(rows, store, test_region, rng):
    """Paired differences in the interior violation rate: noise 0 -> max noise at each size, and smallest -> largest size at each noise."""
    interior = test_region == "interior"
    ones = np.ones(len(test_region), bool)
    out = []
    by = {}
    for r in rows:
        by.setdefault(r["name"], {})[(r["n"], r["noise"])] = r
    for name, cells in by.items():
        sizes, noises = sorted({k[0] for k in cells}), sorted({k[1] for k in cells})
        if len(noises) < 2 and len(sizes) < 2:
            continue
        key_of = {(r["n"], r["noise"]): k for k, st in store.items() for r in [dict(n=k[7], noise=k[8], name=name_of(k))] if r["name"] == name}
        for n in sizes:                              # noise effect at fixed size
            if (n, noises[0]) in key_of and (n, noises[-1]) in key_of and len(noises) > 1:
                ka, kb = key_of[(n, noises[0])], key_of[(n, noises[-1])]
                if store[ka]["flags"].shape != store[kb]["flags"].shape:
                    continue
                ra, rb = boot_rates([store[ka]["flags"], store[kb]["flags"]], [interior, interior], rng)
                out.append(dict(name=name, contrast=f"noise {noises[-1]:g} minus noise {noises[0]:g} at n={n}", region="interior",
                                rate_a=float(store[ka]["flags"][:, interior].mean()), rate_b=float(store[kb]["flags"][:, interior].mean()),
                                diff=interval(rb - ra)))
        for z in noises:                             # size effect at fixed noise
            if (sizes[0], z) in key_of and (sizes[-1], z) in key_of and len(sizes) > 1:
                ka, kb = key_of[(sizes[0], z)], key_of[(sizes[-1], z)]
                if store[ka]["flags"].shape != store[kb]["flags"].shape:
                    continue
                ra, rb = boot_rates([store[ka]["flags"], store[kb]["flags"]], [interior, interior], rng)
                out.append(dict(name=name, contrast=f"n={sizes[0]} minus n={sizes[-1]} at noise {z:g}", region="interior",
                                rate_a=float(store[ka]["flags"][:, interior].mean()), rate_b=float(store[kb]["flags"][:, interior].mean()),
                                diff=interval(ra - rb)))
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def variance_link(store, test_region, truth_sub, rng, name):
    """Spearman correlation, across the (size, noise) settings of one model, between the variance term and the interior violation rate.

    The seed bootstrap resamples the same seeds in every setting, recomputes each setting's variance and interior rate, and the
    correlation. Needs at least 4 settings with the same seeds."""
    keys = [k for k in store if name_of(k) == name]
    if len(keys) < 4:
        return None
    interior = test_region == "interior"
    sub_in = interior[:truth_sub.shape[0]]
    S = min(store[k]["flags"].shape[0] for k in keys)
    if S < 5:
        return None

    def stat(idx):
        v, rate = [], []
        for k in keys:
            p = store[k]["preds"][idx]
            ok = ~np.isnan(truth_sub) & ~np.isnan(p).any(axis=0) & sub_in[:, None]
            v.append(np.mean(p[:, ok].var(axis=0)) if ok.any() else np.nan)
            rate.append(store[k]["flags"][idx][:, interior].mean())
        return spearman(np.array(v), np.array(rate))

    point = stat(np.arange(S))
    boots = np.array([stat(rng.integers(0, S, S)) for _ in range(B_BV)])
    boots = boots[~np.isnan(boots)]
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if len(boots) > 20 else (float("nan"),) * 2
    return dict(name=name, settings=len(keys), seeds=S, spearman=point, lo=ci[0], hi=ci[1])


# ---------------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------------

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [flat(r) if any(isinstance(v, tuple) for v in r.values()) else r for r in rows]
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def md_table(header, body):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in body]) + "\n"


def size_noise_table(rows, name):
    rs = sorted([r for r in rows if r["name"] == name], key=lambda r: (r["n"], r["noise"]))
    if not rs:
        return ""
    body = [[f"{r['n']:,}", f"{100 * r['noise']:g}%", r["seeds"], ci_str(r["rmse_iv"], 1e3, 2),
             ci_str(r["viol_all"], percent=True, region="all", seeds=r["seeds"]), ci_str(r["viol_interior"], percent=True, region="interior", seeds=r["seeds"]),
             ci_str(r["viol_boundary"], percent=True, region="boundary", seeds=r["seeds"]), pct(r["conv_rate"][0]), pct(r["cal_rate"][0]), pct(r["sev_1e4"][0])] for r in rs]
    return md_table(["n", "noise", "seeds", "RMSE (vol, x1e-3)", "violating: all", "interior", "boundary", "conv", "cal", ">1e-4"], body)


def figures(rows, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    for name in sorted({r["name"] for r in rows if r["model"] in ("grid_net", "ridge_poly", "gradient_boosting", "gaussian_process") and r["noise"] in (0.0, 0.02)}):
        rs = [r for r in rows if r["name"] == name]
        sizes = sorted({r["n"] for r in rs})
        noises = sorted({r["noise"] for r in rs})
        if len(noises) < 2 or len(sizes) < 2:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for n in sizes:
            xs = [z for z in noises if any(r["n"] == n and r["noise"] == z for r in rs)]
            for a, field, title in ((ax[0], "viol_interior", "interior surfaces violating"), (ax[1], "rmse_iv", "RMSE (implied vol)")):
                ys = [next(r for r in rs if r["n"] == n and r["noise"] == z)[field] for z in xs]
                a.errorbar([100 * z for z in xs], [y[0] for y in ys], yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]], marker="o", capsize=2, label=f"n={n:,}")
                a.set_title(f"{name}: {title}", fontsize=9); a.set_xlabel("label noise (%)")
        ax[0].set_yscale("symlog", linthresh=1e-4); ax[1].set_yscale("log"); ax[0].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, f"noise_{name.replace(' ', '_')}.png"), dpi=140); plt.close(fig)
    # accuracy against violation: one point per setting
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in sorted({r["model"] for r in rows if r["model"] != "pointwise"}):
        rs = [r for r in rows if r["model"] == model]
        ax.scatter([r["rmse_iv"][0] for r in rs], [max(r["viol_all"][0], 1e-5) for r in rs], label=model, s=18)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("RMSE (implied vol)"); ax.set_ylabel("share of surfaces violating")
    ax.legend(fontsize=7); ax.set_title("Accuracy against consistency, one point per setting", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "accuracy_vs_violation.png"), dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(data.ROOT, "results", "runs"))
    ap.add_argument("--out-dir", default=os.path.join(data.ROOT, "results"))
    ap.add_argument("--docs-out", default=os.path.join(data.ROOT, "docs", "experiments_results.md"))
    a = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)
    groups = load_runs(a.run_dir)
    if not groups:
        sys.exit("no runs found")
    test = data.load_test()
    for k, m in (("all", np.ones(len(test["region"]), bool)), ("interior", test["region"] == "interior"), ("boundary", test["region"] == "boundary")):
        REGION_N[k] = int(m.sum())
    from experiment import N_SUBSET
    truth_sub = test["iv"][:N_SUBSET]
    rows, store = aggregate(groups, test["region"], truth_sub, a.run_dir, rng)
    write_csv(os.path.join(a.out_dir, "tables", "settings.csv"), rows)
    trends = h2_trends(rows, store, test["region"], rng)
    write_csv(os.path.join(a.out_dir, "tables", "h2_trends.csv"), [dict(name=t["name"], contrast=t["contrast"], rate_a=t["rate_a"], rate_b=t["rate_b"],
                                                                         diff=t["diff"][0], diff_lo=t["diff"][1], diff_hi=t["diff"][2]) for t in trends])
    links = [l for l in (variance_link(store, test["region"], truth_sub, rng, nm) for nm in sorted({r["name"] for r in rows})) if l]
    write_csv(os.path.join(a.out_dir, "tables", "variance_link.csv"), links)
    figures(rows, os.path.join(a.out_dir, "figures"))

    md = ["# Experiment results (generated)\n",
          "Generated by `scripts/analyse_experiments.py` from `results/runs/`. Do not edit by hand. Protocol, decisions and interpretation: "
          "`docs/experiments.md`. Intervals are 95 percent bootstrap intervals over seeds (independent training sets), with test "
          "surfaces resampled too for rates. No correction for multiple comparisons.\n"]
    for name in sorted({r["name"] for r in rows}):
        rs = [r for r in rows if r["name"] == name]
        if len({r["n"] for r in rs}) > 1:
            md += [f"\n## {name}: size and noise\n", f"Seeds per setting: {sorted({r['seeds'] for r in rs})}.\n", size_noise_table(rows, name)]
    md += ["\n## H1: do curvature errors predict violating stencils better than pointwise errors?\n",
           "AUC of each local error measure for ranking convexity stencils of the base grid (positive = the model violates convexity "
           "there). Primary comparison: relative curvature error minus pointwise implied vol error. Secondary: absolute curvature error minus "
           "absolute price error. See the caveat in `docs/experiments.md`: a violation requires a large curvature error by construction.\n"]
    body = []
    for r in sorted(rows, key=lambda r: (r["name"], r["n"], r["noise"])):
        if np.isnan(r["auc_curvature_rel"][0]):
            continue
        body.append([r["name"], f"{r['n']:,}", f"{100 * r['noise']:g}%", f"{r['auc_positives_mean']:.0f}", ci_str(r["auc_pointwise_iv"], digits=3),
                     ci_str(r["auc_curvature_rel"], digits=3), ci_str(r["auc_diff_primary"], digits=3), ci_str(r["auc_diff_secondary"], digits=3)])
    md.append(md_table(["model", "n", "noise", "violating stencils / run", "AUC pointwise vol error", "AUC relative curvature error", "primary difference", "secondary difference"], body))
    md += ["\n## H2: paired trends in the interior violation rate\n",
           f"Detector false-positive bound: at most {FPR_PER_SURFACE:.1e} per surface (95 percent, docs/violations.md). A difference whose "
           "interval excludes zero is shown in bold.\n"]
    body = []
    for t in trends:
        d = t["diff"]
        cell = f"{100 * d[0]:+.2f} pts [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]"
        if d[1] > 0 or d[2] < 0:
            cell = f"**{cell}**"
        if d[0] == 0 and d[1] == 0 and d[2] == 0:
            cell = "0 (no violating interior surface in either setting)"
        body.append([t["name"], t["contrast"], pct(t["rate_a"]), pct(t["rate_b"]), cell])
    md.append(md_table(["model", "contrast", "rate a", "rate b", "difference (a minus b for size; b minus a for noise), interior"], body))
    md += ["\n## Bias and variance across seeds\n",
           "Squared bias and variance of the implied vol prediction, averaged over resolved cells of the first test surfaces. MSE = plug-in bias^2 + plug-in "
           "variance exactly; with S seeds the plug-in bias^2 is too large by (unbiased variance)/S, so corrected and unbiased columns are given too. "
           "Variance is across independent training sets.\n"]
    body = [[r["name"], f"{r['n']:,}", f"{100 * r['noise']:g}%", r["seeds"], f"{r['bv_mse']:.2e}", f"{r['bias2']:.2e}", f"{r['variance']:.2e}",
             f"{r['bias2_corrected']:.2e}", f"{r['variance_unbiased']:.2e}", f"{r['var_interior']:.2e}", f"{r['var_boundary']:.2e}"]
            for r in sorted(rows, key=lambda r: (r["name"], r["n"], r["noise"])) if r["seeds"] > 1]
    md.append(md_table(["model", "n", "noise", "seeds", "MSE", "bias^2 (plug-in)", "variance (plug-in)", "bias^2 (corrected)", "variance (unbiased)", "plug-in variance, interior", "plug-in variance, boundary"], body))
    if links:
        md += ["\n### Does the variance term track the interior violation rate?\n",
               "Spearman correlation across the settings of one model, with a seed-bootstrap interval.\n",
               md_table(["model", "settings", "seeds", "Spearman", "95% interval"], [[l["name"], l["settings"], l["seeds"], f"{l['spearman']:+.2f}", f"[{l['lo']:+.2f}, {l['hi']:+.2f}]"] for l in links])]
    arch = [r for r in rows if r["model"] == "grid_net" and r["n"] == 10000 and r["noise"] in (0.0, 0.02)]
    if len({r["name"] for r in arch}) > 1:
        md += ["\n## Architecture: one factor at a time around the base grid network (n = 10,000)\n",
               "Base is 3 layers of 30 ELU units. Seeds: 10 for the base, 5 for the variants.\n"]
        body = [[r["name"].replace("grid_net ", ""), f"{100 * r['noise']:g}%", r["seeds"], ci_str(r["rmse_iv"], 1e3, 2),
                 ci_str(r["viol_all"], percent=True, region="all", seeds=r["seeds"]), ci_str(r["viol_interior"], percent=True, region="interior", seeds=r["seeds"]),
                 ci_str(r["viol_boundary"], percent=True, region="boundary", seeds=r["seeds"]), pct(r["conv_rate"][0]), pct(r["cal_rate"][0]),
                 pct(r["sev_1e4"][0]), f"{r['fit_seconds']:.0f}"] for r in sorted(arch, key=lambda r: (r["noise"], r["name"]))]
        md.append(md_table(["variant", "noise", "seeds", "RMSE (x1e-3)", "violating: all", "interior", "boundary", "conv", "cal", ">1e-4", "fit s"], body))
    pw = [r for r in rows if r["model"] == "pointwise"]
    if pw:
        md += ["\n## Pointwise networks\n"]
        body = [[r["name"], f"{100 * r['noise']:g}%", r["seeds"], ci_str(r["price_rmse"], 1e3, 2), ci_str(r["rmse_iv"], 1e3, 2), ci_str(r["viol_all"], percent=True, region="all", seeds=r["seeds"]), ci_str(r["viol_interior"], percent=True, region="interior", seeds=r["seeds"]),
                 ci_str(r["dense_viol"], percent=True), ci_str(r["neg_density_share"], percent=True), pct(r["unusable"][0])] for r in sorted(pw, key=lambda r: (r["name"], r["noise"]))]
        md.append(md_table(["model", "noise", "seeds", "price RMSE (x1e-3)", "vol RMSE (x1e-3)", "violating: base grid", "interior", "violating: dense grid", "negative autodiff density share", "unusable cells"], body))
    os.makedirs(os.path.dirname(a.docs_out), exist_ok=True)
    with open(a.docs_out, "w") as f:
        f.write("\n".join(md))
    print(f"{len(rows)} settings, {sum(r['seeds'] for r in rows)} runs -> {os.path.relpath(a.docs_out, data.ROOT)}")


if __name__ == "__main__":
    main()
