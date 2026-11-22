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


def _tiny(n=60, seed=0, **kw):
    tr = data.load_train(n, seed)
    dv = data.load_derivs("train", tr["params"], seed=seed)
    cfg = N.TrainConfig(width=32, depth=2, activation="softplus", epochs=25, batch_size=512, schedule="cosine", seed=seed, **kw)
    return N.PointwiseSurrogate(cfg, target="price").fit(tr["params"], tr["call"], tr["iv"], dv), tr, dv


@need_torch
def test_repair_derivatives_agree_with_strike_derivatives():
    m, tr, _ = _tiny(20, deriv_weight=0.1)
    p = tr["params"][:3]
    theta, k, t = m._grid_inputs(p, H.STRIKES, H.MATURITIES)
    price, d1, d2, dt = m._price_and_derivs(theta, N.scale_k(k), N.scale_t(t), density=True, calendar=True)
    c, e1, e2 = m.strike_derivatives(p)
    assert np.allclose(d1.detach().numpy().reshape(e1.shape), e1, atol=1e-10)
    assert np.allclose(d2.detach().numpy().reshape(e2.shape), e2, atol=1e-8)
    assert np.isfinite(dt.detach().numpy()).all()


@need_torch
def test_derivative_loss_improves_density_fit():
    exact = data.load_derivs("train", data.load_train(60, 0)["params"], seed=0)["density"].reshape(60, 8, 11)
    errs = []
    for w in (0.0, 1.0):
        m, tr, _ = _tiny(60, deriv_weight=w)
        _, _, d2 = m.strike_derivatives(tr["params"])
        errs.append(np.sqrt(np.mean((d2 - exact) ** 2)))
    assert errs[1] < errs[0]


@need_torch
def test_penalty_is_zero_on_valid_and_positive_on_violating_surfaces():
    m, tr, _ = _tiny(20, penalty_weight=1.0)
    assert m.history_terms[0][2] > 0                      # an untrained network violates somewhere, and the term sees it
    m.s0, m.s1, m.s2_by_maturity = 1.0, 1.0, np.ones(len(H.MATURITIES))
    theta = torch.zeros(50, 5, dtype=N.DTYPE)
    k = torch.linspace(0.6, 1.4, 50, dtype=N.DTYPE)
    t = torch.linspace(0.2, 1.8, 50, dtype=N.DTYPE)

    class Fixed(torch.nn.Module):             # column 5 is the scaled strike, column 6 the scaled maturity
        def __init__(self, f):
            super().__init__(); self.f = f
        def forward(self, x):
            return self.f(x[:, 5], x[:, 6])[:, None]

    def penalty(f):
        m.net = Fixed(f)
        return float(m.arbitrage_penalty(theta, k, t).detach())
    assert penalty(lambda ks, ts: (1 - ks) * (2 + ts) ** 0.5) == 0.0      # decreasing, convex in strike, increasing in maturity
    assert penalty(lambda ks, ts: ks) > 0                                 # increasing in strike
    assert penalty(lambda ks, ts: -ks ** 2 + 3 * (1 - ks)) > 0            # concave in strike
    assert penalty(lambda ks, ts: (1 - ks) * (2 - ts)) > 0                # decreasing in maturity


@need_torch
def test_repairs_need_price_target_and_labels():
    tr = data.load_train(20, 0)
    for target, deriv, kw in (("iv", {"slope": tr["iv"], "density": tr["iv"]}, dict(deriv_weight=1.0)), ("price", None, dict(penalty_weight=1.0))):
        try:
            N.PointwiseSurrogate(N.TrainConfig(epochs=1, batch_size=512, **kw), target=target).fit(tr["params"], tr["call"], tr["iv"], deriv)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
