"""Checks for the demo (app/): the numpy forward pass equals the trained torch network, and one evaluation runs end to end.
Needs the trained files in app/models (python scripts/train_demo_models.py); skipped if they are missing. Torch is needed only for the first test."""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "app"))
HAVE = os.path.exists(os.path.join(ROOT, "app", "models", "price_pen.npz"))
if HAVE:
    import core, surrogates
    import heston as H


def test_numpy_forward_matches_torch():
    if not HAVE:
        return print("skipped (no models)")
    try:
        import torch
    except ImportError:
        return print("skipped (no torch)")
    sys.path.insert(0, os.path.join(ROOT, "src"))
    w = dict(np.load(os.path.join(ROOT, "app", "models", "price_pen.npz")))
    layers, i = [], 0
    while f"W{i}" in w:
        layers.append(torch.nn.Linear(w[f"W{i}"].shape[1], w[f"W{i}"].shape[0]).double())
        layers[-1].weight.data, layers[-1].bias.data = torch.tensor(w[f"W{i}"]), torch.tensor(w[f"b{i}"])
        i += 1
    net = torch.nn.Sequential(*[m for j, l in enumerate(layers) for m in ((l, torch.nn.Softplus()) if j < len(layers) - 1 else (l,))])
    x = np.random.default_rng(0).uniform(-1, 1, (20, 7))
    assert np.allclose(net(torch.tensor(x)).detach().numpy(), surrogates._mlp(w, x), atol=1e-10)


def test_evaluate_and_search_run():
    if not HAVE:
        return print("skipped (no models)")
    models = core.get_models()
    assert set(models) >= {"grid", "ridge", "price", "price_pen"}
    p = np.array(core.PRESETS["Typical"])
    res = core.evaluate(p, models)
    for name in models:
        assert res[name]["rates"].keys() == set(core.CONDITIONS) and res[name]["iv_rmse"] < 0.2
    assert len(core.table(res)) == len(models)
    core.fig_density(p, res, 2), core.fig_map(res)
    # the exact surface passes the detector on the same dense grid (no false alarms in the demo)
    import violations as V
    assert not any(f.any() for f in V.flags(res["exact"]["dense"][None], H.DENSE_STRIKES, H.DENSE_MATURITIES).values())
    q, worst, mean = core.search_breaking(models["price"][1], n=20)
    assert worst >= mean and H.LB.min() <= q.min() and (q <= H.UB).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
