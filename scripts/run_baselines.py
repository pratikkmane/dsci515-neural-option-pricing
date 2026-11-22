"""Fit a baseline on exact Heston surfaces and evaluate it on the fixed test set (step 4).

Example: python scripts/run_baselines.py --models ridge_poly gradient_boosting --n-train 10000 --seed 0 --noise 0.005

Training labels get multiplicative noise at the chosen level (test labels stay exact). One JSON per run goes to
results/baselines/, named by model, size, noise and seed. One seed is a single draw: quote spread only from several seeds.
"""
import argparse, json, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data, baselines, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(baselines.MODELS), choices=list(baselines.MODELS))
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="selects the training set (Sobol scramble) and the model's randomness")
    ap.add_argument("--noise", type=float, default=0.0, help="multiplicative label noise, e.g. 0.005 for 0.5 percent")
    ap.add_argument("--workers", type=int, default=1, help="processes for generating surfaces")
    ap.add_argument("--out-dir", default=os.path.join(data.ROOT, "results", "baselines"))
    a = ap.parse_args()
    warnings.simplefilter("ignore")

    train = data.load_train(a.n_train, a.seed, a.workers)
    test = data.load_test(a.workers)
    iv_train = data.add_noise(train["iv"], a.noise, seed=10_000 + a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    for name in a.models:
        model = baselines.MODELS[name](seed=a.seed)
        t0 = time.time(); model.fit(train["params"], iv_train); fit_s = time.time() - t0
        t0 = time.time(); pred = model.predict(test["params"]); pred_s = time.time() - t0
        res = dict(model=name, n_train=a.n_train, seed=a.seed, label_noise=a.noise, fit_seconds=fit_s, predict_seconds=pred_s,
                   n_test=len(test["params"]), train_nan_cell_share=float(np.isnan(train["iv"]).mean()),
                   model_info=model.info, results=evaluate.evaluate_grid(pred, test))
        path = os.path.join(a.out_dir, f"{name}_n{a.n_train}_noise{a.noise:g}_seed{a.seed}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=float)
        r = res["results"]["all"]
        print(f"{name:18s} n={a.n_train} noise={a.noise:g} seed={a.seed}: RMSE {r['accuracy']['rmse_iv']:.5f}  "
              f"p99 {r['accuracy']['p99_abs_error_iv']:.5f}  surfaces violating {r['violations']['any_violation_surface_rate']:.3f}  "
              f"(fit {fit_s:.0f}s)  -> {os.path.relpath(path, data.ROOT)}")


if __name__ == "__main__":
    main()
