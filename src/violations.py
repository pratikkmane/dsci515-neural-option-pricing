"""Static no-arbitrage checks for call price surfaces (Carr and Madan, 2005), with zero rates and dividends, spot 1.

A surface is an array of call prices C[n, T, K] on strikes K (ascending) and maturities T (ascending). Each condition is
the price of a portfolio whose payoff can never be negative, so a "violation margin" is the price of that portfolio when it
is negative, in price units (the money that could be locked in per unit of the portfolio):

  monotone    call spread      C(K_i) - C(K_{i+1}) >= 0                     margin = C(K_{i+1}) - C(K_i)
  convexity   butterfly        w1 C(K_{i-1}) - C(K_i) + w3 C(K_{i+1}) >= 0  margin = minus that price
                               with w1 = (K_{i+1} - K_i) / (K_{i+1} - K_{i-1}), w3 = (K_i - K_{i-1}) / (K_{i+1} - K_{i-1})
  calendar    time spread      C(K, T_{j+1}) - C(K, T_j) >= 0               margin = C(K, T_j) - C(K, T_{j+1})

A stencil is flagged when its margin exceeds a tolerance. Because margins are prices, the same tolerance serves any grid:
exact prices carry numerical noise of about 1e-13 (see docs/violations.md for how the default was set).

The convexity margin is tied to probability: with slopes s_i = (C_{i+1} - C_i) / (K_{i+1} - K_i), the slope increment
s_i - s_{i-1} is the risk-neutral probability mass around K_i, and butterfly price = increment * h+ h- / (h+ + h-).
`negative_mass` adds up the negative increments, in probability units.

NaN inputs (for example unresolved implied vols) give NaN margins, which are never flagged and are not counted as checked.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr

CONDITIONS = ("monotone", "convexity", "calendar")


@dataclass(frozen=True)
class Tolerance:
    """Smallest margin (price units) counted as a violation, per condition."""
    monotone: float
    convexity: float
    calendar: float

    @classmethod
    def uniform(cls, value):
        return cls(value, value, value)


# Set by scripts/calibrate_violation_tolerance.py (10 x the largest exact-surface margin, 2.3e-14, rounded up to a power
# of ten). Zero false positives in about 31 million checks on fresh exact surfaces: docs/violations.md.
DEFAULT_TOL = Tolerance.uniform(1e-12)


def iv_to_call(iv, strikes, maturities):
    """Black-Scholes call prices (forward 1) from implied vols on a grid. iv: (n, T*K) maturity-major or (n, T, K)."""
    strikes = np.asarray(strikes, float)
    maturities = np.asarray(maturities, float)
    iv = np.asarray(iv, float)
    iv = iv.reshape(len(iv), len(maturities), len(strikes))
    sd = iv * np.sqrt(maturities)[None, :, None]
    k = strikes[None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -np.log(k) / sd + 0.5 * sd
        return ndtr(d1) - k * ndtr(d1 - sd)


def margins(call, strikes, maturities):
    """Violation margins (positive = violated), NaN where an input is NaN.

    Returns {"monotone": (n, T, K-1), "convexity": (n, T, K-2), "calendar": (n, T-1, K)}.
    """
    call = np.asarray(call, float)
    k = np.asarray(strikes, float)
    if call.ndim != 3 or call.shape[1:] != (len(maturities), len(k)):
        raise ValueError(f"call must have shape (n, {len(maturities)}, {len(k)}), got {call.shape}")
    if np.any(np.diff(k) <= 0) or np.any(np.diff(np.asarray(maturities, float)) <= 0):
        raise ValueError("strikes and maturities must be strictly increasing")
    hm, hp = k[1:-1] - k[:-2], k[2:] - k[1:-1]
    big_h = hm + hp
    butterfly = (hp / big_h) * call[..., :-2] - call[..., 1:-1] + (hm / big_h) * call[..., 2:]
    return {
        "monotone": call[..., 1:] - call[..., :-1],
        "convexity": -butterfly,
        "calendar": call[:, :-1, :] - call[:, 1:, :],
    }


def flags(call, strikes, maturities, tol=DEFAULT_TOL):
    """Boolean violation flags per stencil (same shapes as `margins`)."""
    tol = tol if isinstance(tol, Tolerance) else Tolerance.uniform(float(tol))
    m = margins(call, strikes, maturities)
    return {c: m[c] > getattr(tol, c) for c in CONDITIONS}


def point_flags(call, strikes, maturities, tol=DEFAULT_TOL):
    """Per grid point, shape (n, T, K): True if any violated stencil involves that point.

    Convexity flags the centre strike, a monotone flag marks both strikes of the spread, a calendar flag both maturities.
    This is the cell-level label to use when asking whether an error measure predicts violations (H1).
    """
    f = flags(call, strikes, maturities, tol)
    out = np.zeros(np.asarray(call).shape, bool)
    out[..., 1:-1] |= f["convexity"]
    out[..., :-1] |= f["monotone"]
    out[..., 1:] |= f["monotone"]
    out[:, :-1, :] |= f["calendar"]
    out[:, 1:, :] |= f["calendar"]
    return out


def negative_mass(call, strikes, maturities, tol=DEFAULT_TOL):
    """Negative implied probability mass per surface and maturity, shape (n, T), in probability units.

    Sum of -(s_i - s_{i-1}) over interior strikes whose convexity violation exceeds the tolerance.
    """
    tol = tol if isinstance(tol, Tolerance) else Tolerance.uniform(float(tol))
    call = np.asarray(call, float)
    k = np.asarray(strikes, float)
    slope = np.diff(call, axis=2) / np.diff(k)
    increment = np.diff(slope, axis=2)                       # s_i - s_{i-1}, centred on interior strikes
    m = margins(call, strikes, maturities)["convexity"]
    return np.where(m > tol.convexity, -increment, 0.0).sum(axis=2)


SEVERITY_THRESHOLDS = (1e-8, 1e-6, 1e-4)      # price units, spot 1: 1e-4 is one basis point of the spot


def summarise(call, strikes, maturities, tol=DEFAULT_TOL, severity=SEVERITY_THRESHOLDS):
    """Violation statistics per condition: surface rate, cell rate, size, negative probability mass, and severity.

    `severity` gives, for each threshold, the share of surfaces whose largest violation margin (any condition) exceeds it,
    so that a rate based on the 1e-12 tolerance can be read next to how large the violations are."""
    tol = tol if isinstance(tol, Tolerance) else Tolerance.uniform(float(tol))
    m = margins(call, strikes, maturities)
    n = len(m["monotone"])
    out, any_bad = {}, np.zeros(n, bool)
    for c in CONDITIONS:
        x = m[c]
        valid = ~np.isnan(x)
        bad = valid & (x > getattr(tol, c))
        bad_surface = bad.reshape(n, -1).any(axis=1)
        any_bad |= bad_surface
        checked = valid.reshape(n, -1).any(axis=1)
        out[c] = dict(
            tolerance=getattr(tol, c),
            checks=int(valid.sum()),
            surface_rate=float(bad_surface[checked].mean()) if checked.any() else float("nan"),
            cell_rate=float(bad.sum() / valid.sum()) if valid.any() else float("nan"),
            max_margin=float(np.nanmax(x)) if valid.any() else float("nan"),
            mean_margin_when_violated=float(x[bad].mean()) if bad.any() else 0.0,
        )
    out["negative_mass"] = dict(mean_per_surface=float(np.nanmean(negative_mass(call, strikes, maturities, tol).sum(axis=1))))
    out["any_violation_surface_rate"] = float(any_bad.mean())
    worst = np.nanmax(np.stack([np.nanmax(m[c].reshape(n, -1), axis=1) for c in CONDITIONS]), axis=0) if n else np.array([])
    out["severity_surface_rate"] = {f"{t:.0e}": float(np.mean(worst > t)) for t in severity}
    return out
