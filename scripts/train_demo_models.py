"""Train the models shown in the demo and export them as small .npz files in app/models/ (numpy only, no torch needed at demo time).

  grid         the base-paper network (ELU, 30 x 3, 10,000 surfaces, 200 epochs)
  price        pointwise price network, squared-error loss only (softplus, 64 x 3, 5,000 surfaces, 60 epochs, cosine)
  price_pen    the same with the arbitrage penalty, weight 10 (the weight that reached the 1 percent target in step 6)
  ridge        polynomial ridge on the parameters (10,000 surfaces)
All use seed 0 and exact labels (no noise): this is one example network of each kind, the evidence is in results/, not here.
Usage: python scripts/train_demo_models.py [name ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import data

OUT = os.path.join(data.ROOT, "app", "models")


def linear_layers(net):
    import torch
    out = {}
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        out[f"W{i}"], out[f"b{i}"] = m.weight.detach().numpy(), m.bias.detach().numpy()
    return out


def train(name):
    import networks as N, baselines
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if name == "grid":
        tr = data.load_train(10000, 0)
        m = N.GridSurrogate(N.TrainConfig(epochs=200, batch_size=N.grid_batch_size(10000), seed=0)).fit(tr["params"], tr["iv"])
        np.savez(os.path.join(OUT, "grid.npz"), activation="elu", mean=m.std.mean, std=m.std.std, **linear_layers(m.net))
    elif name in ("price", "price_pen"):
        tr = data.load_train(5000, 0)
        w = 10.0 if name == "price_pen" else 0.0
        dv = data.load_derivs("train", tr["params"], seed=0)
        cfg = N.TrainConfig(width=64, depth=3, activation="softplus", epochs=60, batch_size=1024, schedule="cosine", seed=0, penalty_weight=w)
        m = N.PointwiseSurrogate(cfg, target="price").fit(tr["params"], tr["call"], tr["iv"], dv)
        np.savez(os.path.join(OUT, name + ".npz"), activation="softplus", **linear_layers(m.net))
    elif name == "ridge":
        tr = data.load_train(10000, 0)
        m = baselines.RidgePoly(seed=0).fit(tr["params"], tr["iv"])
        np.savez(os.path.join(OUT, "ridge.npz"), degree=m.info["degree"], m=m.m, s=m.s, coef=m.coef, ym=m.ym, mean=m.surf.std.mean, std=m.surf.std.std)
    else:
        raise SystemExit(f"unknown model {name}")
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["ridge", "grid", "price", "price_pen"]:
        train(n)
