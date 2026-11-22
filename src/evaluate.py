"""Evaluation of a model that outputs the 88-point implied vol surface (step 4).

Accuracy is measured on cells where the exact surface is resolved; unresolved cells (NaN) are masked, never filled. The
violation check converts the predicted implied vols to call prices (Black-Scholes, forward 1) and applies src/violations.py on
the base grid. Predicted vols that are NaN or not positive cannot be turned into prices: they are counted in
`unusable_cell_share` and left out of the violation checks (so a model cannot hide failures by predicting nonsense that is
silently skipped: the share is reported next to the rates).

Results are given overall and for the interior and boundary parts of the test set (src/data.py).
"""
import numpy as np
import heston as H
import violations as V

MIN_VOL = 1e-4


def accuracy(pred, true):
    """RMSE, 99th percentile absolute error and relative error of implied vol over cells where both are available."""
    ok = ~np.isnan(true) & ~np.isnan(pred)
    e = (pred - true)[ok]
    rel = 100 * np.abs(e / true[ok])
    by_mat = []
    for j in range(len(H.MATURITIES)):
        cols = slice(j * len(H.STRIKES), (j + 1) * len(H.STRIKES))
        m = ok[:, cols]
        by_mat.append(float(np.sqrt(np.mean(((pred - true)[:, cols][m]) ** 2))) if m.any() else float("nan"))
    return dict(
        rmse_iv=float(np.sqrt(np.mean(e ** 2))),
        p99_abs_error_iv=float(np.quantile(np.abs(e), 0.99)),
        mean_rel_error_pct=float(rel.mean()),
        max_rel_error_pct=float(rel.max()),
        rmse_iv_by_maturity=by_mat,
        cells_compared=int(ok.sum()),
    )


def predicted_call_surface(pred, true):
    """Call prices from predicted vols on cells where the exact surface is resolved; NaN elsewhere.

    Returns (call (n, T, K), unusable_cell_share): the share of resolved cells whose predicted vol is NaN or not positive."""
    resolved = ~np.isnan(true)
    usable = resolved & ~np.isnan(pred) & (pred > MIN_VOL)
    iv = np.where(usable, pred, np.nan)
    call = V.iv_to_call(iv, H.STRIKES, H.MATURITIES)
    return call, float(1 - usable.sum() / resolved.sum())


def violations(pred, true):
    """src/violations.py summary on the base grid, plus the share of resolved cells that could not be checked."""
    call, unusable = predicted_call_surface(pred, true)
    s = V.summarise(call, H.STRIKES, H.MATURITIES)
    s["unusable_cell_share"] = unusable
    return s


def evaluate_grid(pred, test):
    """Accuracy and violations overall and by region. `test` is data.load_test() output."""
    out = {}
    for label, sel in [("all", np.ones(len(pred), bool))] + [(r, test["region"] == r) for r in ("interior", "boundary")]:
        out[label] = dict(n_surfaces=int(sel.sum()), accuracy=accuracy(pred[sel], test["iv"][sel]),
                          violations=violations(pred[sel], test["iv"][sel]))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Models that output call prices (pointwise networks)
# ---------------------------------------------------------------------------------------------------------------------

def prices_to_iv(call, true):
    """Implied vols (n, 88) from predicted call prices (n, T, K); NaN where the exact surface is unresolved or inversion fails.

    A predicted out-of-the-money price below PRICE_FLOOR (including a negative one) cannot be inverted: it becomes NaN and is
    counted as unusable by `evaluate_prices`.
    """
    k, t = H.STRIKES[None, None, :], H.MATURITIES[None, :, None]
    otm = np.where(k >= 1, call, call - (1 - k))
    iv = H.implied_vol_otm(otm, k, t).reshape(len(call), -1)
    return np.where(np.isnan(true), np.nan, iv)


def evaluate_prices(call, test):
    """Accuracy and violations for predicted base-grid call prices (n, T, K), overall and by region.

    Violations are checked on the prices directly, on cells where the exact surface is resolved. Accuracy is in implied vol on
    cells where the predicted price can be inverted; the share that cannot is `unusable_cell_share`. `price_rmse` is over all cells.
    """
    out = {}
    exact = test["call"].reshape(call.shape)
    for label, sel in [("all", np.ones(len(call), bool))] + [(r, test["region"] == r) for r in ("interior", "boundary")]:
        c, true_iv = call[sel], test["iv"][sel]
        resolved = ~np.isnan(true_iv).reshape(c.shape)
        iv = prices_to_iv(c, true_iv)
        s = V.summarise(np.where(resolved, c, np.nan), H.STRIKES, H.MATURITIES)
        s["unusable_cell_share"] = float(1 - (~np.isnan(iv)).sum() / resolved.sum())
        out[label] = dict(n_surfaces=int(sel.sum()), accuracy=accuracy(iv, true_iv), violations=s,
                          price_rmse=float(np.sqrt(np.mean((c - exact[sel]) ** 2))))
    return out


def curvature_report(d2, tol=0.0):
    """Share of (surface, maturity, strike) points whose autodiff density d2C/dK2 is below -tol, and the mean negative density.

    d2 is the autodiff second strike derivative (n, T, K) from PointwiseSurrogate.strike_derivatives. A net with ReLU units has
    d2 = 0 almost everywhere, so this reports no negative density for it: use the finite-difference checks as well.
    """
    neg = d2 < -tol
    return dict(negative_density_share=float(neg.mean()),
                mean_negative_density=float(d2[neg].mean()) if neg.any() else 0.0,
                surfaces_with_negative_density=float(neg.reshape(len(d2), -1).any(axis=1).mean()))


# ---------------------------------------------------------------------------------------------------------------------
# Local errors as predictors of violating stencils (H1)
# ---------------------------------------------------------------------------------------------------------------------

def auc(score, label):
    """Area under the ROC curve (Mann-Whitney) of `score` for the boolean `label`; NaN if either class is empty."""
    from scipy.stats import rankdata
    label = np.asarray(label, bool)
    pos = int(label.sum())
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _butterfly(c):
    """Butterfly price at every interior strike of a (n, T, K) array on the base grid."""
    k = H.STRIKES
    hm, hp = k[1:-1] - k[:-2], k[2:] - k[1:-1]
    big = hm + hp
    return (hp / big) * c[..., :-2] - c[..., 1:-1] + (hm / big) * c[..., 2:]


def local_error_auc(pred_call, pred_iv, test, tol=None, min_positive=30):
    """How well does a local error measure rank the stencils that violate convexity? (H1)

    A stencil is a convexity stencil of the base grid (an interior strike at one maturity); it is positive if the predicted
    surface violates convexity there (margin above the tolerance). Each measure is computed from the predicted and exact
    surfaces on the stencil's three points:
      pointwise_price   largest absolute call price error
      pointwise_iv      largest absolute implied vol error (NaN where the predicted vol is unusable: those stencils are dropped)
      curvature_abs     absolute error of the butterfly price (the discrete strike curvature)
      curvature_rel     that error divided by the exact butterfly price (floored at the tolerance): the relative curvature error
    Pre-specified primary comparison (docs/experiments.md): curvature_abs against pointwise_price.
    NOTE the link by construction: a violation needs the butterfly error to be negative and larger than the exact butterfly
    price, so a large curvature error is necessary for a violation. The comparison asks whether the curvature error
    carries more information than the level error, not whether the two are unrelated.

    pred_call (n, T, K) may hold NaN (unusable); pred_iv (n, 88) may too. Stencils with any missing exact or predicted value
    are dropped. Returns dict(auc per measure, n_stencils, n_positive); the AUCs are NaN below `min_positive` positives.
    """
    tol = V.DEFAULT_TOL.convexity if tol is None else tol
    exact_call = test["call"].reshape(pred_call.shape)
    resolved = ~np.isnan(test["iv"]).reshape(pred_call.shape)
    pc = np.where(resolved, pred_call, np.nan)
    b_pred, b_true = _butterfly(pc), _butterfly(np.where(resolved, exact_call, np.nan))
    label = (-b_pred) > tol
    ok = ~np.isnan(b_pred) & ~np.isnan(b_true)

    def three(x):            # elementwise stencil stack of an (n, T, K) array -> (3, n, T, K-2)
        return np.stack([x[..., :-2], x[..., 1:-1], x[..., 2:]])

    err_price = np.abs(three(pc) - three(exact_call)).max(axis=0)
    iv_err = np.abs(pred_iv.reshape(pred_call.shape) - test["iv"].reshape(pred_call.shape))
    measures = {
        "pointwise_price": err_price,
        "pointwise_iv": three(iv_err).max(axis=0),
        "curvature_abs": np.abs(b_pred - b_true),
        "curvature_rel": np.abs(b_pred - b_true) / np.maximum(b_true, tol),
    }
    out = {"n_stencils": int(ok.sum()), "n_positive": int((label & ok).sum()), "auc": {}}
    for name, m in measures.items():
        use = ok & ~np.isnan(m)
        lab = label[use]
        out["auc"][name] = auc(m[use], lab) if lab.sum() >= min_positive else float("nan")
    return out


def violating_surfaces(call):
    """Boolean per surface: any Carr and Madan condition violated at the default tolerance (NaN stencils ignored)."""
    f = V.flags(call, H.STRIKES, H.MATURITIES)
    return np.logical_or.reduce([x.reshape(len(call), -1).any(axis=1) for x in f.values()])
