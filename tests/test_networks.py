"""Fast checks for src/networks.py and the price-output evaluation. Run with `python tests/test_networks.py`.

Needs torch; without it every test is skipped (the rest of the suite does not need torch).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import heston as H
    import data, evaluate as E, networks as N


def need_torch(fn):
    def wrapper():
        if torch is None:
            print("skipped (torch not installed):", fn.__name__)
            return
        fn()
    wrapper.__name__ = fn.__name__
    return wrapper


def _data(n, seed):
    p = H.sobol_params(n, seed=seed)
    iv, call = data.surfaces(p)
    return p, iv, call


@need_torch
def test_masked_loss_ignores_missing_cells():
    pred = torch.tensor([[1.0, 5.0]], dtype=N.DTYPE)
    y = torch.tensor([[2.0, 0.0]], dtype=N.DTYPE)
    assert np.isclose(N.masked_rmse(pred, y, torch.tensor([[1.0, 0.0]], dtype=N.DTYPE)).item(), 1.0)


@need_torch
def test_grid_surrogate_learns_and_is_deterministic():
    p, iv, _ = _data(300, 31)
    cfg = N.TrainConfig(epochs=25, batch_size=32, seed=5)
    a = N.GridSurrogate(cfg).fit(p, iv)
    b = N.GridSurrogate(cfg).fit(p, iv)
    assert a.history[-1] < 0.5 * a.history[0]
    assert np.array_equal(a.predict(p[:20]), b.predict(p[:20]))               # same seed, same network
    assert a.predict(p[:20]).shape == (20, 88)
    assert not np.array_equal(a.predict(p[:20]), N.GridSurrogate(N.TrainConfig(epochs=25, seed=6)).fit(p, iv).predict(p[:20]))


@need_torch
def test_autodiff_density_matches_black_scholes():
    """Constant-vol network output: autograd d2C/dK2 must equal phi(d2) / (K sigma sqrt(T)), which checks every scaling."""
    sigma = 0.25
    s = N.PointwiseSurrogate(N.TrainConfig(width=4, depth=1, activation="softplus"), target="iv")
    s.scale_iv = 1.0
    s.net = N.mlp(7, 1, 4, 1, "softplus")
    with torch.no_grad():
        for prm in s.net.parameters():
            prm.zero_()
        s.net[-1].bias.fill_(np.log(np.expm1(sigma - 1e-6)))                   # softplus(bias) + 1e-6 = sigma
    p = H.sobol_params(2, seed=1)
    price, d1, d2 = s.strike_derivatives(p)
    k, t = H.STRIKES[None, :], H.MATURITIES[:, None]
    from scipy.special import ndtr
    sd = sigma * np.sqrt(t)
    dd1 = -np.log(k) / sd + 0.5 * sd
    dd2 = dd1 - sd
    assert np.allclose(price[0], ndtr(dd1) - k * ndtr(dd2), atol=1e-12)
    assert np.allclose(d1[0], -ndtr(dd2), atol=1e-10)                          # dC/dK = -N(d2)
    assert np.allclose(d2[0], np.exp(-0.5 * dd2 ** 2) / np.sqrt(2 * np.pi) / (k * sd), atol=1e-9)


@need_torch
def test_autodiff_second_derivative_matches_finite_difference():
    s = N.PointwiseSurrogate(N.TrainConfig(width=12, depth=2, activation="softplus"), target="price")
    torch.manual_seed(0)
    s.net = N.mlp(7, 1, 12, 2, "softplus")
    p = H.sobol_params(3, seed=2)
    ks = np.array([0.8, 1.0, 1.2]); h = 1e-3
    _, _, d2 = s.strike_derivatives(p, ks, [1.0])
    f = lambda k: s.predict_call(p, np.array([k]), [1.0])[:, 0, 0]
    fd = np.column_stack([(f(k + h) - 2 * f(k) + f(k - h)) / h ** 2 for k in ks])
    assert np.allclose(d2[:, 0, :], fd, rtol=1e-4, atol=1e-6)


@need_torch
def test_relu_price_net_has_zero_curvature_but_finite_differences_see_kinks():
    s = N.PointwiseSurrogate(N.TrainConfig(width=12, depth=2, activation="relu"), target="price")
    torch.manual_seed(1)
    s.net = N.mlp(7, 1, 12, 2, "relu")
    p = H.sobol_params(4, seed=3)
    _, _, d2 = s.strike_derivatives(p, H.DENSE_STRIKES, H.DENSE_MATURITIES)
    assert np.abs(d2).max() == 0.0
    assert E.curvature_report(d2)["negative_density_share"] == 0.0


@need_torch
def test_pointwise_surrogate_trains_on_prices_and_on_vols():
    p, iv, call = _data(150, 41)
    for target in ("price", "iv"):
        m = N.PointwiseSurrogate(N.TrainConfig(width=20, depth=2, epochs=6, batch_size=256, seed=2, activation="softplus"), target=target)
        m.fit(p, call, iv)
        assert m.history[-1] < m.history[0]
        pred = m.predict_call(p[:5])
        assert pred.shape == (5, 8, 11) and np.isfinite(pred).all()
        assert m.predict_call(p[:5], H.DENSE_STRIKES, H.DENSE_MATURITIES).shape == (5, 20, 101)


@need_torch
def test_evaluate_prices_on_exact_prices():
    p = H.sobol_params(60, seed=51)
    iv, call = data.surfaces(p)
    test = dict(params=p, iv=iv, call=call, region=data.region(p))
    r = E.evaluate_prices(call.reshape(-1, 8, 11).copy(), test)["all"]
    assert r["price_rmse"] == 0.0 and r["violations"]["any_violation_surface_rate"] == 0.0
    assert r["violations"]["unusable_cell_share"] < 0.01 and r["accuracy"]["rmse_iv"] < 1e-6
    bad = call.reshape(-1, 8, 11).copy(); bad[:, :, 0] = -0.01                  # absurd prices: counted as unusable
    assert E.evaluate_prices(bad, test)["all"]["violations"]["unusable_cell_share"] > 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
