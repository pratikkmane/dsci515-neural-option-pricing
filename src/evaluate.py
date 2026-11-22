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
