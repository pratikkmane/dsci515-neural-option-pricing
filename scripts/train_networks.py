"""Train a neural surrogate on exact Heston surfaces and evaluate it on the fixed test set (step 4).

Examples:
  python scripts/train_networks.py --kind grid --n-train 10000 --epochs 200 --seed 0
  python scripts/train_networks.py --kind pointwise --target price --activation softplus --n-train 10000 --epochs 20

Training labels get multiplicative noise (test labels stay exact). Grid models are evaluated on the 88-point base grid.
Pointwise models are evaluated on the base grid and, in addition, on the dense grid (101 strikes x 20 maturities) for the first
--dense-n test surfaces, with the autodiff strike curvature. One JSON per run goes to results/models/ (weights to *.pt, ignored
by git). One seed is a single draw: quote spread only from several seeds.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data, evaluate, networks
import heston as H
import violations as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["grid", "pointwise"], default="grid")
    ap.add_argument("--target", choices=["price", "iv"], default="price", help="pointwise output: call price or implied vol")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--activation", choices=list(networks.ACTIVATIONS), default="elu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dense-n", type=int, default=2000, help="test surfaces used for the dense-grid and autodiff checks")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--out-dir", default=os.path.join(data.ROOT, "results", "models"))
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    default = networks.TrainConfig(batch_size=32, epochs=200) if a.kind == "grid" else networks.TrainConfig(batch_size=1024, epochs=30)
    cfg = networks.TrainConfig(width=a.width, depth=a.depth, activation=a.activation, lr=a.lr, seed=a.seed,
                               epochs=a.epochs or default.epochs, batch_size=a.batch_size or default.batch_size)
    train, test = data.load_train(a.n_train, a.seed, a.workers), data.load_test(a.workers)
    noise_seed = 10_000 + a.seed
    iv = data.add_noise(train["iv"], a.noise, noise_seed)
    call = data.add_noise(train["call"], a.noise, noise_seed + 1)

    t0 = time.time()
    if a.kind == "grid":
        model = networks.GridSurrogate(cfg).fit(train["params"], iv)
    else:
        model = networks.PointwiseSurrogate(cfg, target=a.target).fit(train["params"], call, iv)
    fit_s = time.time() - t0

    t0 = time.time()
    res = dict(kind=a.kind, target=a.target if a.kind == "pointwise" else "iv", n_train=a.n_train, seed=a.seed, label_noise=a.noise,
               fit_seconds=fit_s, model_info=model.info, n_test=len(test["params"]))
    if a.kind == "grid":
        res["results"] = evaluate.evaluate_grid(model.predict(test["params"]), test)
    else:
        call_test = model.predict_call(test["params"])
        res["results"] = evaluate.evaluate_prices(call_test, test)
        sub = test["params"][:a.dense_n]
        _, d1, d2 = model.strike_derivatives(sub, H.DENSE_STRIKES, H.DENSE_MATURITIES)
        dense = model.predict_call(sub, H.DENSE_STRIKES, H.DENSE_MATURITIES)
        res["dense_grid"] = dict(n_surfaces=len(sub), violations=V.summarise(dense, H.DENSE_STRIKES, H.DENSE_MATURITIES),
                                 autodiff_curvature=evaluate.curvature_report(d2),
                                 positive_strike_slope_share=float((d1 > 0).mean()))
    res["evaluate_seconds"] = time.time() - t0

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.kind}{'_' + a.target if a.kind == 'pointwise' else ''}_{a.activation}_w{a.width}d{a.depth}_n{a.n_train}_noise{a.noise:g}_seed{a.seed}"
    with open(os.path.join(a.out_dir, tag + ".json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    torch.save((model.net if hasattr(model, "net") else None).state_dict(), os.path.join(a.out_dir, tag + ".pt"))
    r = res["results"]["all"]
    line = (f"{tag}: RMSE_iv {r['accuracy']['rmse_iv']:.5f}  p99 {r['accuracy']['p99_abs_error_iv']:.5f}  "
            f"surfaces violating (base grid) {r['violations']['any_violation_surface_rate']:.3f}  unusable {r['violations']['unusable_cell_share']:.4f}  "
            f"fit {fit_s:.0f}s")
    if a.kind == "pointwise":
        dg = res["dense_grid"]
        line += (f"\n    dense grid: surfaces violating {dg['violations']['any_violation_surface_rate']:.3f}  "
                 f"negative autodiff density share {dg['autodiff_curvature']['negative_density_share']:.3f}")
    print(line)


if __name__ == "__main__":
    main()
