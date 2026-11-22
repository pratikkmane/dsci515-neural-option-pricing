"""Choose the pointwise networks' training budget on the VALIDATION set (never the test set).

Trains the same pointwise network with several budgets on one training set and reports validation accuracy, violation rate and
time, so that the budget is fixed before any pointwise result on the test set is looked at. Output: results/pilot_pointwise.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data, evaluate, networks


def run(spec):
    name, target, kw, n = spec
    torch.set_num_threads(1)
    train, val = data.load_train(n, 0), data.load_val()
    cfg = networks.TrainConfig(seed=0, batch_size=1024, **kw)
    t0 = time.time()
    m = networks.PointwiseSurrogate(cfg, target=target).fit(train["params"], train["call"], train["iv"])
    fit = time.time() - t0
    r = evaluate.evaluate_prices(m.predict_call(val["params"]), val)["all"]
    return dict(name=name, target=target, n_train=n, config=kw, fit_seconds=fit, final_train_loss=m.history[-1],
                val_rmse_iv=r["accuracy"]["rmse_iv"], val_price_rmse=r["price_rmse"], unusable_cell_share=r["violations"]["unusable_cell_share"],
                val_violating_surfaces=r["violations"]["any_violation_surface_rate"], val_severity=r["violations"]["severity_surface_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--targets", nargs="+", default=["iv", "price"])
    ap.add_argument("--processes", type=int, default=4)
    a = ap.parse_args()
    data.load_train(a.n_train, 0, 4); data.load_val(4)            # generate once before the processes start
    grid = [("w64d3_e40_const", dict(width=64, depth=3, epochs=40, activation="softplus")),
            ("w64d3_e40_cos", dict(width=64, depth=3, epochs=40, activation="softplus", schedule="cosine")),
            ("w128d4_e40_cos", dict(width=128, depth=4, epochs=40, activation="softplus", schedule="cosine")),
            ("w128d4_e100_cos", dict(width=128, depth=4, epochs=100, activation="softplus", schedule="cosine"))]
    specs = [(n, t, kw, a.n_train) for t in a.targets for n, kw in grid]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.processes) as ex:
        results = list(ex.map(run, specs))
    for r in results:
        print(f"{r['target']:5s} {r['name']:16s} val RMSE_iv {r['val_rmse_iv']:.5f}  price RMSE {r['val_price_rmse']:.5f}  unusable {r['unusable_cell_share']:.3f}  "
              f"violating {r['val_violating_surfaces']:.3f}  fit {r['fit_seconds']:.0f}s")
    os.makedirs("results", exist_ok=True)
    with open("results/pilot_pointwise.json", "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
