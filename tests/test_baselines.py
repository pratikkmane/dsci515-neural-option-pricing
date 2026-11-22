"""Fast checks for src/data.py, src/evaluate.py and src/baselines.py. Run with `python tests/test_baselines.py`."""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import heston as H
import data, evaluate as E, baselines as B


def test_standardiser_ignores_nan():
    y = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    s = data.Standardiser().fit(y)
    assert np.allclose(s.mean, [2.0, 0.0]) and np.allclose(s.std, [1.0, 1.0])     # empty column falls back to (0, 1)
    z = s.transform(y)
    assert np.allclose(s.inverse(z)[~np.isnan(y)], y[~np.isnan(y)])


def test_noise():
    y = np.full((20000, 3), 0.2); y[0, 0] = np.nan
    assert np.array_equal(data.add_noise(y, 0.0, 1), y, equal_nan=True)
    n = data.add_noise(y, 0.01, 1)
    assert np.isnan(n[0, 0]) and abs(np.nanstd(n / 0.2) - 0.01) < 5e-4
    assert np.array_equal(data.add_noise(y, 0.01, 1), n, equal_nan=True)           # same seed, same noise
    assert not np.array_equal(data.add_noise(y, 0.01, 2), n, equal_nan=True)


def test_region_labels():
    mid = (H.LB + H.UB) / 2
    edge = mid.copy(); edge[2] = H.LB[2] + 0.05 * (H.UB[2] - H.LB[2])
    assert list(data.region(np.array([mid, edge]))) == ["interior", "boundary"]


def test_larger_cached_set_is_sliced():
    old = data.GEN_DIR
    with tempfile.TemporaryDirectory() as d:
        data.GEN_DIR = d
        try:
            big = data.load_train(256, seed=1)
            small = data.load_train(100, seed=1)                                        # read from the 256-row cache
            assert not os.path.exists(os.path.join(d, "train_n100_seed1.npz"))
            assert np.array_equal(small["iv"], big["iv"][:100], equal_nan=True)
            assert len(data.load_val(n=64)["params"]) == 64
        finally:
            data.GEN_DIR = old


def test_nested_training_sets_and_cache():
    old = data.GEN_DIR
    with tempfile.TemporaryDirectory() as d:
        data.GEN_DIR = d
        try:
            a = data.load_train(256, seed=3)
            b = data.load_train(128, seed=3)
            assert np.array_equal(a["params"][:128], b["params"])                           # parameters: exactly nested
            assert np.allclose(a["iv"][:128], b["iv"], atol=1e-9, equal_nan=True)       # vols: equal up to batch-size rounding
            assert not np.array_equal(data.load_train(128, seed=4)["params"], b["params"])
            again = data.load_train(256, seed=3)                                    # second call reads the cache
            assert os.path.exists(os.path.join(d, "train_n256_seed3.npz")) and np.array_equal(again["iv"], a["iv"], equal_nan=True)
        finally:
            data.GEN_DIR = old


def test_complete_low_rank_recovers_missing_cells():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((400, 3)) @ rng.standard_normal((3, 12))
    holes = rng.random(z.shape) < 0.1
    filled = B.complete_low_rank(np.where(holes, np.nan, z), rank=3, iters=60)
    assert np.allclose(filled[~holes], z[~holes]) and np.abs(filled[holes] - z[holes]).max() < 1e-2


def test_accuracy_known_values():
    true = np.array([[0.2, 0.3, np.nan] + [0.2] * 85])
    pred = true + 0.01
    pred[0, 2] = 0.5                                                              # where truth is NaN: ignored
    a = E.accuracy(pred, true)
    assert np.isclose(a["rmse_iv"], 0.01) and a["cells_compared"] == 87


def test_exact_surfaces_pass_evaluation():
    p = H.sobol_params(20, seed=8)
    iv, _ = data.surfaces(p)
    v = E.violations(iv.copy(), iv)
    assert v["any_violation_surface_rate"] == 0.0 and v["unusable_cell_share"] == 0.0
    bad = iv.copy(); bad[:, 3] = np.where(np.isnan(iv[:, 3]), np.nan, -0.1)       # non-positive vols are reported, not hidden
    assert E.violations(bad, iv)["unusable_cell_share"] > 0


def test_baselines_fit_and_predict():
    tr = data.surfaces(H.sobol_params(600, seed=21))[0]
    p_tr = H.sobol_params(600, seed=21)
    p_te = H.sobol_params(200, seed=22)
    iv_te = data.surfaces(p_te)[0]
    ok = ~np.isnan(iv_te)
    for model in (B.RidgePoly(degrees=(2, 3, 4)), B.PCAGradientBoosting(n_components=6, max_iter=60),
                  B.PCAGaussianProcess(n_components=4, max_train=150, n_hyper=100, restarts=0)):
        pred = model.fit(p_tr, tr).predict(p_te)
        assert pred.shape == (200, 88)
        rmse = np.sqrt(np.mean((pred - iv_te)[ok] ** 2))
        assert rmse < 0.05, (model.name, rmse)
        assert isinstance(model.info, dict)
    assert B.PCAGaussianProcess(n_components=4, max_train=150, n_hyper=100, restarts=0).fit(p_tr, tr).info["capped"] is True


def test_auc_known_values_and_local_error_auc():
    assert E.auc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert E.auc(np.array([0.4, 0.3, 0.2, 0.1]), np.array([0, 0, 1, 1], bool)) == 0.0
    assert E.auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 1, 0, 1], bool)) == 0.5          # ties give one half
    assert np.isnan(E.auc(np.array([1.0, 2.0]), np.array([0, 0], bool)))
    # exact surfaces plus a few injected dents: the butterfly-error measure must rank the dented stencils first
    p = H.sobol_params(200, seed=71)
    iv, call = data.surfaces(p)
    test = dict(params=p, iv=iv, call=call, region=data.region(p))
    pred = call.reshape(-1, 8, 11).copy()
    rng = np.random.default_rng(0)
    s_idx, t_idx, k_idx = rng.integers(0, 200, 60), rng.integers(0, 8, 60), rng.integers(1, 10, 60)
    pred[s_idx, t_idx, k_idx] += 5e-3
    pred_iv = E.prices_to_iv(pred, iv)
    r = E.local_error_auc(pred, pred_iv, test)
    assert r["n_positive"] > 30 and r["auc"]["curvature_abs"] > 0.95 and r["auc"]["pointwise_price"] > 0.95


def test_refit_recomputes_component_count_and_ridge_guard():
    p1 = H.sobol_params(300, seed=61); p2 = H.sobol_params(300, seed=62)
    m = B.PCAGradientBoosting(max_iter=20)
    m.fit(p1, data.surfaces(p1)[0])
    k1 = m.k
    noisy = data.surfaces(p2)[0] * (1 + 0.05 * np.random.default_rng(0).standard_normal((300, 88)))
    m.fit(p2, noisy)                                                  # a noisier surface needs more components for the same target
    assert m.k != k1 and m.k == m.surf.choose_k(noisy, m.target, m.k_max)   # recomputed, not the stale first value
    try:
        B.RidgePoly().fit(p1[:20], data.surfaces(p1[:20])[0])
    except ValueError as e:
        assert "too few training rows" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
