"""Logic of the demo (no UI): exact and model surfaces for one parameter set, violation checks, figures, and the search for a breaking setting.

Works from the repository (imports ../src) and from the deployed Space (imports ./src, a copy of heston.py and violations.py).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if os.path.isdir(d):
        sys.path.insert(0, d)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import heston as H
import violations as V
import surrogates

COLOURS = {"exact": "black", "grid": "tab:orange", "ridge": "tab:green", "price": "tab:red", "price_pen": "tab:blue"}
CONDITIONS = ("monotone", "convexity", "calendar")
NAMES = {"monotone": "decreasing in strike", "convexity": "convex in strike", "calendar": "increasing in maturity"}
PARAM_LABELS = ["Initial variance xi0", "Correlation rho", "Vol of vol sigma", "Long-run variance theta", "Mean reversion kappa"]
PRESETS = {"Typical": [0.02, -0.5, 0.4, 0.06, 4.0], "Low initial variance, high vol of vol": [0.0004, -0.9, 0.9, 0.04, 8.0],
           "Calm market": [0.01, -0.3, 0.1, 0.02, 2.0]}


def get_models():
    return surrogates.load_all(H.LB, H.UB, H.STRIKES, H.MATURITIES)


def _base_call_and_iv(model, p):
    """(call (8, 11), iv (8, 11)) of a model on the base grid; call is NaN where the model gives no usable price."""
    k, t = H.STRIKES[None, :], H.MATURITIES[:, None]
    if model.kind == "grid":
        iv = model.predict_iv(p)[0].reshape(len(H.MATURITIES), len(H.STRIKES))
        iv = np.where(iv > 1e-4, iv, np.nan)
        return V.iv_to_call(iv[None], H.STRIKES, H.MATURITIES)[0], iv
    call = model.predict_call(p, H.STRIKES, H.MATURITIES)
    otm = np.where(k >= 1, call, call - (1 - k))
    return call, H.implied_vol_otm(otm, k, t)


def evaluate(p, models):
    """Everything the UI shows for one parameter set. Returns dict with 'exact' and one entry per model."""
    p = np.asarray(p, float)
    exact_iv = H.implied_vols(p[None])[0]
    exact_base = H.call_prices(p[None])[0]
    out = {"exact": dict(base=exact_base, iv=exact_iv, dense=H.call_prices(p[None], H.DENSE_STRIKES, H.DENSE_MATURITIES)[0])}
    for name, (label, m) in models.items():
        base, iv = _base_call_and_iv(m, p)
        ok = ~np.isnan(exact_iv) & ~np.isnan(iv)
        r = dict(label=label, base=base, iv_rmse=float(np.sqrt(np.mean((iv - exact_iv)[ok] ** 2))) if ok.any() else float("nan"),
                 price_rmse=float(np.sqrt(np.nanmean((base - exact_base) ** 2))))
        if m.kind == "pointwise":
            r["grid"] = (H.DENSE_STRIKES, H.DENSE_MATURITIES)
            r["call"] = m.predict_call(p, *r["grid"])
        else:
            r["grid"] = (H.STRIKES, H.MATURITIES)
            r["call"] = np.where(~np.isnan(exact_iv), base, np.nan)         # unresolved exact cells are not checked (as in the experiments)
        k, t = r["grid"]
        r["flags"] = V.flags(r["call"][None], k, t)
        r["points"] = V.point_flags(r["call"][None], k, t)[0]
        m_ = V.margins(r["call"][None], k, t)
        r["rates"] = {c: float(np.mean(r["flags"][c][0][~np.isnan(m_[c][0])])) for c in CONDITIONS}
        r["worst_margin"] = {c: float(np.nanmax(m_[c])) for c in CONDITIONS}
        out[name] = r
    return out


def table(res):
    rows = []
    for name, r in res.items():
        if name == "exact":
            continue
        k, _ = r["grid"]
        rows.append([r["label"], f"{len(k)}-strike grid", f"{100 * r['iv_rmse']:.2f}", f"{1e3 * r['price_rmse']:.2f}"] +
                    [f"{100 * r['rates'][c]:.1f}" for c in CONDITIONS])
    return rows


TABLE_HEADER = ["Model", "Checked on", "Implied vol RMSE (vol points)", "Price RMSE (1e-3)",
                "Violating checks: not decreasing in strike (%)", "not convex in strike (%)", "not increasing in maturity (%)"]


def fig_density(p, res, maturity_index):
    """Risk-neutral density d2C/dK2 against strike at one maturity: exact (analytic) and each model (second differences)."""
    t = float(H.MATURITIES[maturity_index])
    _, _, d2 = H.call_strike_derivatives(np.asarray(p)[None], H.DENSE_STRIKES, [t])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax.plot(H.DENSE_STRIKES, d2[0, 0], color="black", lw=2, label="exact")
    for name, r in res.items():
        if name == "exact":
            continue
        k, mats = r["grid"]
        j = int(np.argmin(abs(np.asarray(mats) - t)))
        c = r["call"][j]
        h = np.diff(k)
        dens = (c[2:] - 2 * c[1:-1] + c[:-2]) / (0.5 * (h[1:] + h[:-1])) ** 2 if np.allclose(h, h[0]) else None
        if dens is None:
            continue
        style = dict(color=COLOURS[name], label=r["label"])
        (ax.plot(k[1:-1], dens, lw=1.2, **style) if len(k) > 20 else ax.plot(k[1:-1], dens, "o-", ms=4, lw=1, **style))
        neg = np.minimum(dens, 0)
        ax2.plot(k[1:-1], neg, "o-" if len(k) < 20 else "-", ms=4, lw=1.2, **style)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Implied density at maturity {t:g}")
    ax.set_xlabel("strike")
    ax.set_ylabel("d2C/dK2 (must be at least 0)")
    ax.legend(fontsize=7)
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_title("Only the negative part (negative probability)")
    ax2.set_xlabel("strike")
    fig.tight_layout()
    return fig


def fig_map(res):
    """Where each model breaks a rule: violating points over (strike, maturity)."""
    names = [n for n in res if n != "exact"]
    fig, axes = plt.subplots(1, max(len(names), 1), figsize=(3.6 * max(len(names), 1), 3.4), squeeze=False)
    for ax, n in zip(axes[0], names):
        r = res[n]
        k, t = r["grid"]
        ax.pcolormesh(k, t, np.where(np.isnan(r["call"]), 0.5, r["points"].astype(float)), cmap="Reds", vmin=0, vmax=1.4, shading="nearest")
        ax.set_title(f"{r['label']}\n{100 * r['points'].mean():.1f}% of points involved", fontsize=8)
        ax.set_xlabel("strike")
        ax.set_ylabel("maturity")
    fig.tight_layout()
    return fig


def search_breaking(model, n=300, seed=0):
    """Among n random parameter sets, the one where `model` violates the most checks on the base grid. Returns (params, share, mean share)."""
    cand = H.sobol_params(n, seed=seed)
    shares, exact = [], H.implied_vols(cand)
    for p, iv in zip(cand, exact):
        call, _ = _base_call_and_iv(model, p)
        call = np.where(~np.isnan(iv), call, np.nan)[None]
        f = V.flags(call, H.STRIKES, H.MATURITIES)
        m = V.margins(call, H.STRIKES, H.MATURITIES)
        bad = sum(int(f[c].sum()) for c in CONDITIONS)
        tot = sum(int((~np.isnan(m[c])).sum()) for c in CONDITIONS)
        shares.append(bad / tot)
    shares = np.array(shares)
    return cand[int(np.argmax(shares))], float(shares.max()), float(shares.mean())
