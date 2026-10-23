"""Exact Heston (1993) European call pricer, Sobol parameter sampler and implied volatility inversion.

Conventions (checked against data/HestonTrainSet.txt.gz, see scripts/validate_heston.py):
  spot S0 = 1, interest rate and dividend yield 0 (so the forward is 1 and strike K means K/S0),
  parameters in the order (xi0, rho, sigma, theta, kappa) with xi0 the initial variance, sigma the volatility
  of variance, theta the long-run VARIANCE (not volatility) and kappa the mean-reversion speed.

Method: Lewis / Gatheral Fourier representation
    call = 1 - sqrt(K) / pi * integral_0^inf Re[exp(i u ln K) phi(u - i/2)] / (u^2 + 1/4) du,
with the 'little Heston trap' form of the characteristic function phi (Albrecher et al., 2007), written so that
a - d is never computed as a difference of nearly equal numbers. The integral is evaluated by composite
Gauss-Legendre quadrature. The integration range is chosen per parameter set and maturity (the integrand decays
very slowly when xi0 is tiny and sigma is large), and parameter sets are grouped into buckets of equal range so
everything stays vectorised.

Out-of-the-money prices are used for the implied volatility (puts below the forward, calls above) because the
in-the-money call is dominated by intrinsic value. Prices below PRICE_FLOOR cannot be resolved in double
precision by this integral and give NaN implied volatility; the `reliable` mask marks them.

Only numpy and scipy are needed.
"""
import warnings
import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc

PARAM_NAMES = ["xi0", "rho", "sigma", "theta", "kappa"]
LB = np.array([0.0001, -0.95, 0.01, 0.01, 1.0])   # same bounds as src/reproduce_heston.py (base code)
UB = np.array([0.04, -0.1, 1.0, 0.2, 10.0])
STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])      # base-paper grid
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
# Denser grid for checking convexity and calendar monotonicity. It contains every base strike and maturity.
DENSE_STRIKES = np.round(np.linspace(0.5, 1.5, 101), 10)
DENSE_MATURITIES = np.round(np.linspace(0.1, 2.0, 20), 10)

PRICE_FLOOR = 1e-11     # smallest out-of-the-money price treated as resolved (see validate_heston.py)
_PANEL = 4.0            # panel width in u; at |ln K| <= 0.7 the phase advances at most 2.8 rad per panel
_NODES = 16             # Gauss-Legendre nodes per panel
_HEAD_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
_TAIL_TOL = 1e-17       # integrand bound at the truncation point, times the point itself
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_NODES)


def sobol_params(n, seed=0, lb=LB, ub=UB):
    """n parameter sets from a scrambled Sobol sequence, scaled to [lb, ub]. Returns shape (n, 5)."""
    m = max(1, int(np.ceil(np.log2(n))))
    u = qmc.Sobol(d=len(lb), scramble=True, seed=seed).random_base2(m)[:n]
    return lb + u * (ub - lb)


def _phi(z, t, xi0, rho, sigma, theta, kappa):
    """E[exp(i z log(S_T / S_0))] for complex z; parameter arrays broadcast against z."""
    a = kappa - 1j * rho * sigma * z
    d = np.sqrt(a * a + sigma ** 2 * (z * z + 1j * z))
    amd = -sigma ** 2 * (z * z + 1j * z) / (a + d)            # a - d, without cancellation
    g = amd / (a + d)
    e = np.exp(-d * t)
    # log((1 - g e) / (1 - g)) = log1p(g (1 - e) / (1 - g)), accurate when g is small
    log_term = np.log1p(g * (1 - e) / (1 - g))
    h = (1 - e) / (1 - g * e)
    return np.exp(kappa * theta / sigma ** 2 * (amd * t - 2 * log_term) + xi0 / sigma ** 2 * amd * h)


def _integrand_scale(u, t, p):
    """|phi(u - i/2)| / (u^2 + 1/4) for one u per parameter set (p has shape (n, 5))."""
    return np.abs(_phi(u - 0.5j, t, *p.T)) / (u * u + 0.25)


def _truncation(t, p):
    """Smallest U in 64 * 2^m (m = 0..) so that the integrand is negligible beyond U, per parameter set."""
    u = np.full(len(p), 64.0)
    for _ in range(14):
        bad = ~(_integrand_scale(u, t, p) * u < _TAIL_TOL)
        if not bad.any():
            return u
        u[bad] *= 2
    warnings.warn(f"{int(bad.sum())} parameter sets at maturity {t}: integrand not negligible at u = {u.max():.0f}; "
                  "prices for them may be inaccurate")
    return u


def _lewis_integral(p, t, log_k, chunk_nodes=2_000_000):
    """sqrt(K)/pi * integral, shape (len(p), len(log_k)); call price = 1 - this, put price = K - this."""
    out = np.empty((len(p), len(log_k)))
    u_max = _truncation(t, p)
    for U in np.unique(u_max):
        rows = np.flatnonzero(u_max == U)
        # narrow panels near 0 because 1 / (u^2 + 1/4) has poles at u = +-i/2; then constant width _PANEL
        edges = np.concatenate([_HEAD_EDGES, np.arange(2 * _PANEL, U, _PANEL), [U]])   # U >= 64, multiple of 4
        mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        u = (mid[:, None] + half[:, None] * _GL_X[None, :]).ravel()
        w = (half[:, None] * _GL_W[None, :]).ravel() / (u * u + 0.25)
        e_mat = np.exp(1j * np.outer(u, log_k)) * w[:, None]               # (nodes, strikes)
        step = max(1, chunk_nodes // len(u))
        for s in range(0, len(rows), step):
            r = rows[s:s + step]
            phi = _phi(u[None, :] - 0.5j, t, *[c[:, None] for c in p[r].T])  # (rows, nodes)
            out[r] = (phi @ e_mat).real
    return np.sqrt(np.exp(log_k)) / np.pi * out


def otm_prices(params, strikes=STRIKES, maturities=MATURITIES):
    """Out-of-the-money option prices, shape (n, len(maturities), len(strikes)): put if K < 1, else call."""
    params = np.atleast_2d(np.asarray(params, dtype=float))
    strikes = np.asarray(strikes, dtype=float)
    log_k = np.log(strikes)
    out = np.empty((len(params), len(maturities), len(strikes)))
    for j, t in enumerate(maturities):
        x = _lewis_integral(params, float(t), log_k)
        out[:, j, :] = np.where(strikes >= 1, 1 - x, strikes - x)
    return out


def call_prices(params, strikes=STRIKES, maturities=MATURITIES):
    """European call prices (S0 = 1, r = 0), shape (n, len(maturities), len(strikes))."""
    strikes = np.asarray(strikes, dtype=float)
    otm = otm_prices(params, strikes, maturities)
    return np.where(strikes >= 1, otm, otm + 1 - strikes)   # call = put + 1 - K


def _bs_otm(sig, k, t):
    """Black-Scholes out-of-the-money price (forward 1): put if k < 1 else call."""
    sd = sig * np.sqrt(t)
    d1 = -np.log(k) / sd + 0.5 * sd
    d2 = d1 - sd
    return np.where(k >= 1, ndtr(d1) - k * ndtr(d2), k * ndtr(-d2) - ndtr(-d1))


def implied_vol_otm(price, k, t, iters=56):
    """Black-Scholes implied volatility from out-of-the-money prices by vectorised bisection.

    price, k, t broadcast against each other. Returns NaN where the price is below PRICE_FLOOR or outside the
    range attainable with volatility in [1e-3, 4]."""
    price, k, t = np.broadcast_arrays(np.asarray(price, float), np.asarray(k, float), np.asarray(t, float))
    lo = np.full(price.shape, 1e-3)
    hi = np.full(price.shape, 4.0)
    ok = (price >= PRICE_FLOOR) & (_bs_otm(lo, k, t) <= price) & (_bs_otm(hi, k, t) >= price)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = _bs_otm(mid, k, t) < price
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def implied_vols(params, strikes=STRIKES, maturities=MATURITIES):
    """Implied volatility surfaces, shape (n, len(maturities), len(strikes)); NaN where unresolved.

    Flatten the last two axes (maturity-major) to get the 88-vector layout of the shipped data."""
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    price = otm_prices(params, strikes, maturities)
    return implied_vol_otm(price, strikes[None, None, :], maturities[None, :, None])


def surface88(params):
    """The base-paper target: (n, 88) implied vols, maturity-major (8 maturities x 11 strikes)."""
    return implied_vols(params).reshape(len(np.atleast_2d(params)), -1)
