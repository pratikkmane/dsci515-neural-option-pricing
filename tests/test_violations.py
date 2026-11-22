"""Fast checks for src/violations.py. Run with `python tests/test_violations.py` (or pytest)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import heston as H
import violations as V

K = np.linspace(0.5, 1.5, 11)          # uniform spacing h = 0.1
T = np.array([0.5, 1.0, 2.0])


def bs_surface(vol=0.2):
    sd = vol * np.sqrt(T)[:, None]
    d1 = -np.log(K)[None, :] / sd + 0.5 * sd
    from scipy.special import ndtr
    return (ndtr(d1) - K[None, :] * ndtr(d1 - sd))[None]


def linear_surface():
    """C = 1 - K + 0.01 T: zero butterfly price, call spreads of -h, rising by 0.005 or more per maturity step."""
    return ((1 - K)[None, :] + 0.01 * T[:, None])[None]


def test_black_scholes_surface_has_no_violations():
    c = bs_surface()
    assert all(not f.any() for f in V.flags(c, K, T).values())
    m = V.margins(c, K, T)
    assert m["monotone"].max() < 0 and m["calendar"].max() < 0 and m["convexity"].max() < 0


def test_convexity_dent_margin_and_mass():
    c = linear_surface()
    delta, j, i = 1e-4, 1, 5
    c[0, j, i] += delta
    m = V.margins(c, K, T)["convexity"]
    assert np.isclose(m[0, j, i - 1], delta)                          # margin of the dented stencil is the dent
    assert np.isclose(m[0, j, i - 2], -delta / 2) and np.isclose(m[0, j, i], -delta / 2)
    f = V.flags(c, K, T)
    assert f["convexity"].sum() == 1 and f["convexity"][0, j, i - 1]
    assert not f["monotone"].any() and not f["calendar"].any()         # delta is far below a spread of 0.1
    mass = V.negative_mass(c, K, T)
    assert np.isclose(mass[0, j], 2 * delta / 0.1) and mass[0, [0, 2]].sum() == 0


def test_point_flags_mark_the_centre_point():
    c = linear_surface()
    c[0, 1, 5] += 1e-4
    p = V.point_flags(c, K, T)
    assert p.sum() == 1 and p[0, 1, 5]


def test_monotone_violation():
    c = bs_surface()
    c[0, 0, 4:] += 0.5                                                  # price jumps up with strike at index 4
    f = V.flags(c, K, T)
    assert f["monotone"][0, 0, 3] and not f["monotone"][0, 1].any()
    m = V.margins(c, K, T)["monotone"][0, 0, 3]
    assert m > 0.4


def test_calendar_violation():
    c = bs_surface()
    c[0, 2, :] = c[0, 1, :] - 0.01                                      # longer maturity cheaper than shorter
    f = V.flags(c, K, T)
    assert f["calendar"][0, 1].all() and not f["calendar"][0, 0].any()
    assert np.allclose(V.margins(c, K, T)["calendar"][0, 1], 0.01)


def test_tolerance_is_respected():
    c = linear_surface()
    c[0, 1, 5] += 5e-13
    assert not V.flags(c, K, T, tol=1e-12)["convexity"].any()
    assert V.flags(c, K, T, tol=1e-13)["convexity"].any()


def test_nan_is_not_flagged_and_not_counted():
    c = bs_surface()
    c[0, 0, 3] = np.nan
    f = V.flags(c, K, T)
    assert not any(x.any() for x in f.values())
    s = V.summarise(c, K, T)
    assert s["convexity"]["checks"] == 3 * 9 - 3 and s["monotone"]["checks"] == 3 * 10 - 2


def test_summarise_rates():
    c = np.concatenate([linear_surface(), linear_surface()])
    c[1, 1, 5] += 1e-4
    s = V.summarise(c, K, T)
    assert s["convexity"]["surface_rate"] == 0.5 and s["any_violation_surface_rate"] == 0.5
    assert np.isclose(s["convexity"]["cell_rate"], 1 / (2 * 3 * 9))


def test_severity_rates():
    c = np.concatenate([linear_surface()] * 3)
    c[1, 1, 5] += 1e-7                                                  # tiny violation (butterfly margin 1e-7)
    c[2, 1, 5] += 1e-3                                                  # large violation
    s = V.summarise(c, K, T)["severity_surface_rate"]
    assert np.isclose(s["1e-08"], 2 / 3) and np.isclose(s["1e-06"], 1 / 3) and np.isclose(s["1e-04"], 1 / 3)


def test_nonuniform_strikes_use_weighted_butterfly():
    k = np.array([0.5, 0.6, 0.9, 1.0, 1.5])
    c = (1 - k)[None, None, :]
    c = c.copy(); c[0, 0, 2] += 1e-3
    expected = 1e-3                                                       # linear surface: butterfly price is minus the dent
    m = V.margins(c, k, np.array([1.0]))["convexity"][0, 0, 1]
    assert np.isclose(m, expected)


def test_exact_heston_surfaces_have_no_violations_at_default_tolerance():
    p = H.sobol_params(30, seed=9)
    c = H.call_prices(p, H.DENSE_STRIKES, H.DENSE_MATURITIES)
    assert all(not f.any() for f in V.flags(c, H.DENSE_STRIKES, H.DENSE_MATURITIES).values())
    iv = H.implied_vols(p)
    c2 = V.iv_to_call(iv, H.STRIKES, H.MATURITIES)
    assert all(not f.any() for f in V.flags(c2, H.STRIKES, H.MATURITIES).values())


def test_iv_to_call_round_trip():
    p = H.sobol_params(10, seed=3)
    iv = H.implied_vols(p)
    c = V.iv_to_call(iv.reshape(10, -1), H.STRIKES, H.MATURITIES)
    ref = H.call_prices(p)
    ok = ~np.isnan(c)
    assert ok.mean() > 0.95 and np.abs(c[ok] - ref[ok]).max() < 1e-10


def test_bad_shapes_raise():
    for bad in (np.zeros((2, 3)), np.zeros((1, 2, 11))):
        try:
            V.margins(bad, K, T)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    try:
        V.margins(linear_surface(), K[::-1], T)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for descending strikes")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
