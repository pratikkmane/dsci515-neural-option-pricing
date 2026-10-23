"""Fast checks for src/heston.py. Run with `python tests/test_heston.py` (or pytest)."""
import os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import heston as H


def test_martingale():
    # E[S_T / S_0] = 1, i.e. the characteristic function at z = -i equals 1
    p = H.sobol_params(50, seed=1)
    for t in (0.1, 2.0):
        phi = H._phi(np.full(50, -1j), t, *p.T)
        assert np.allclose(phi, 1.0, atol=1e-12)


def test_black_scholes_limit():
    # xi0 = theta and a tiny vol of vol: Heston is close to Black-Scholes with variance xi0 (the skew is of order 0.1 * sigma)
    xi0 = 0.04
    p = np.array([[xi0, -0.5, 0.0005, xi0, 2.0]])
    t, k = np.array([0.5, 1.0]), np.array([0.8, 1.0, 1.2])
    iv = H.implied_vols(p, k, t)
    assert np.allclose(iv, np.sqrt(xi0), atol=1e-4)


def test_implied_vol_roundtrip():
    s = np.array([0.05, 0.2, 0.7, 1.5])
    for k in (0.7, 1.0, 1.3):
        for t in (0.1, 1.0):
            price = H._bs_otm(s, k, t)
            iv = H.implied_vol_otm(price, k, t)
            resolved = price >= H.PRICE_FLOOR            # below the floor the answer is NaN by design
            assert resolved.any() and np.allclose(iv[resolved], s[resolved], atol=1e-9)
            assert np.isnan(iv[~resolved]).all()


def test_surface_is_arbitrage_free_on_dense_grid():
    # exact prices must be decreasing and convex in strike and increasing in maturity, up to the numerical noise floor
    p = H.sobol_params(40, seed=2)
    c = H.call_prices(p, H.DENSE_STRIKES, H.DENSE_MATURITIES)
    d1 = np.diff(c, axis=2)
    d2 = np.diff(c, 2, axis=2)
    dt = np.diff(c, axis=1)
    assert d1.max() < 1e-10 and d2.min() > -1e-10 and dt.min() > -1e-10


def test_shapes_and_grid_layout():
    p = H.sobol_params(5, seed=0)
    assert H.surface88(p).shape == (5, 88)
    assert set(H.STRIKES) <= set(H.DENSE_STRIKES) and set(H.MATURITIES) <= set(H.DENSE_MATURITIES)
    assert H.sobol_params(1000, seed=0).shape == (1000, 5)
    s = H.sobol_params(1000, seed=0)
    assert (s >= H.LB).all() and (s <= H.UB).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
