"""One experiment run, and the named plans that make up the grid (step 5).

A run trains one model on exact Heston surfaces (training set = nested Sobol sample chosen by the seed, multiplicative label noise
on the training labels), evaluates it on the fixed test set, and stores:
  results/runs/<tag>.json   summary: accuracy and violations (overall, interior, boundary), severity, local-error AUC (H1)
  results/runs/<tag>.npz    per-surface violation flags for all test surfaces, and predicted implied vols and call prices for the
                            first N_SUBSET test surfaces (for the bias-variance decomposition). Not committed (regenerable).
The JSON is written last, so a run with a JSON is complete and is skipped when the plan is re-run.
"""
import json, os, time
from dataclasses import dataclass, asdict
import numpy as np

import data, evaluate, baselines
import heston as H

N_SUBSET = 2000        # test surfaces whose predictions are kept for bias and variance
N_DENSE = 500          # test surfaces for the dense-grid and autodiff checks of pointwise networks
RUN_DIR = os.path.join(data.ROOT, "results", "runs")
SIZES = (2000, 10000, 50000, 250000)
NOISES = data.NOISE_LEVELS


@dataclass(frozen=True)
class Run:
    model: str                    # grid_net | pointwise | ridge_poly | gradient_boosting | gaussian_process
    n: int
    noise: float
    seed: int
    width: int = 30
    depth: int = 3
    activation: str = "elu"
    target: str = "iv"            # pointwise output: "iv" or "price"
    epochs: int = 0               # 0 means the default for the model
    batch_size: int = 0           # 0 means the default for the model and size
    lr: float = 1e-3
    schedule: str = "constant"
    deriv_weight: float = 0.0     # step 6: weight of the derivative loss (pointwise price networks)
    deriv_terms: str = "both"     # "both" (slope and density) or "slope"
    penalty_weight: float = 0.0   # step 6: weight of the arbitrage penalty

    def tag(self):
        parts = [self.model]
        if self.model == "pointwise":
            parts.append(self.target)
        if self.model in ("grid_net", "pointwise"):
            parts += [self.activation, f"w{self.width}d{self.depth}"]
            if self.epochs:
                parts.append(f"e{self.epochs}")
            if self.schedule != "constant":
                parts.append(self.schedule)
        if self.deriv_weight:
            parts.append(("dslope" if self.deriv_terms == "slope" else "deriv") + f"{self.deriv_weight:g}")
        if self.penalty_weight:
            parts.append(f"pen{self.penalty_weight:g}")
        parts += [f"n{self.n}", f"noise{self.noise:g}", f"seed{self.seed}"]
        return "_".join(parts)


def _net_config(r):
    import networks as N
    rep = dict(deriv_weight=r.deriv_weight, deriv_terms=r.deriv_terms, penalty_weight=r.penalty_weight)
    if r.model == "grid_net":
        return N.TrainConfig(width=r.width, depth=r.depth, activation=r.activation, lr=r.lr, seed=r.seed, schedule=r.schedule,
                             epochs=r.epochs or 200, batch_size=r.batch_size or N.grid_batch_size(r.n))
    return N.TrainConfig(width=r.width, depth=r.depth, activation=r.activation, lr=r.lr, seed=r.seed, schedule=r.schedule,
                         epochs=r.epochs or POINTWISE_EPOCHS, batch_size=r.batch_size or 1024, **rep)


POINTWISE_EPOCHS = 30     # fallback only; the pointwise plan always sets epochs from POINTWISE_BUDGET


def run_one(r, out_dir=RUN_DIR, workers=1, force=False):
    """Train, evaluate and store one run. Returns the summary dict (read back from disk if the run already exists)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, r.tag() + ".json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    train, test = data.load_train(r.n, r.seed, workers), data.load_test(workers)
    iv_train = data.add_noise(train["iv"], r.noise, 10_000 + r.seed)
    call_train = data.add_noise(train["call"], r.noise, 10_001 + r.seed)

    t0 = time.time()
    extra = {}
    if r.model in baselines.MODELS:
        model = baselines.MODELS[r.model](seed=r.seed).fit(train["params"], iv_train)
        pred_iv = model.predict(test["params"])
        info = model.info
    elif r.model == "grid_net":
        import networks as N
        model = N.GridSurrogate(_net_config(r)).fit(train["params"], iv_train)
        pred_iv = model.predict(test["params"])
        info = model.info
    elif r.model == "pointwise":
        import networks as N
        deriv = None
        if r.deriv_weight or r.penalty_weight:      # exact strike derivatives with the same multiplicative noise as the prices
            dv = data.load_derivs("train", train["params"], seed=r.seed, workers=workers)
            deriv = dict(slope=data.add_noise(dv["slope"], r.noise, 10_002 + r.seed),
                         density=data.add_noise(dv["density"], r.noise, 10_003 + r.seed))
        model = N.PointwiseSurrogate(_net_config(r), target=r.target).fit(train["params"], call_train, iv_train, deriv)
        pred_call_raw = model.predict_call(test["params"])
        info = model.info
    else:
        raise ValueError(r.model)
    fit_s = time.time() - t0

    t0 = time.time()
    true_iv = test["iv"]
    if r.model == "pointwise":
        results = evaluate.evaluate_prices(pred_call_raw, test)
        pred_iv = evaluate.prices_to_iv(pred_call_raw, true_iv)
        resolved = ~np.isnan(true_iv).reshape(pred_call_raw.shape)
        pred_call = np.where(resolved, pred_call_raw, np.nan)
        sub = test["params"][:N_DENSE]
        _, d1, d2 = model.strike_derivatives(sub, H.DENSE_STRIKES, H.DENSE_MATURITIES)
        dense = model.predict_call(sub, H.DENSE_STRIKES, H.DENSE_MATURITIES)
        import violations as V
        _, b1, b2 = model.strike_derivatives(sub)                                  # base grid, against the exact derivatives
        ex = data.load_derivs("test", sub, workers=workers)
        scale2 = np.sqrt(np.mean(ex["density"].reshape(b2.shape) ** 2, axis=(0, 2)))[None, :, None]
        extra["shape_error"] = dict(
            n_surfaces=len(sub),
            slope_rmse=float(np.sqrt(np.mean((b1 - ex["slope"].reshape(b1.shape)) ** 2))),
            density_rmse_scaled=float(np.sqrt(np.mean(((b2 - ex["density"].reshape(b2.shape)) / scale2) ** 2))),
            density_median_abs_rel_error=float(np.median(np.abs(b2 - ex["density"].reshape(b2.shape)) / np.maximum(ex["density"].reshape(b2.shape), 1e-3))))
        extra["dense_grid"] = dict(n_surfaces=len(sub), violations=V.summarise(dense, H.DENSE_STRIKES, H.DENSE_MATURITIES),
                                   autodiff_curvature=evaluate.curvature_report(d2),
                                   positive_strike_slope_share=float((d1 > 0).mean()))
    else:
        results = evaluate.evaluate_grid(pred_iv, test)
        pred_call, _ = evaluate.predicted_call_surface(pred_iv, true_iv)
    auc = evaluate.local_error_auc(pred_call, pred_iv, test)
    viol = evaluate.violating_surfaces(pred_call)
    eval_s = time.time() - t0

    np.savez_compressed(os.path.join(out_dir, r.tag() + ".npz"), viol=viol,
                        pred_iv=pred_iv[:N_SUBSET].astype(np.float32), pred_call=pred_call[:N_SUBSET].reshape(N_SUBSET, -1).astype(np.float32))
    summary = dict(run=asdict(r), tag=r.tag(), fit_seconds=fit_s, evaluate_seconds=eval_s, model_info=info, n_test=len(test["params"]),
                   train_nan_cell_share=float(np.isnan(train["iv"]).mean()), results=results, local_error_auc=auc, **extra)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    os.replace(tmp, path)
    return summary


# ---------------------------------------------------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------------------------------------------------

def plan(name, seeds=10):
    S = range(seeds)
    if name == "core_small":          # size x noise, base-paper network and ridge, sizes up to 50k
        return ([Run("grid_net", n, z, s) for n in SIZES[:3] for z in NOISES for s in S] +
                [Run("ridge_poly", n, z, s) for n in SIZES[:3] for z in NOISES for s in S])
    if name == "core_large":          # the 250k size
        return ([Run("grid_net", SIZES[3], z, s) for z in NOISES for s in S] +
                [Run("ridge_poly", SIZES[3], z, s) for z in NOISES for s in S])
    if name == "boosting":
        return [Run("gradient_boosting", n, z, s) for n in SIZES[:3] for z in (0.0, 0.02) for s in S]
    if name == "gp":                  # capped at 3,000 training points, so only two sizes and few seeds
        return [Run("gaussian_process", n, z, s) for n in SIZES[:2] for z in (0.0, 0.02) for s in range(3)]
    if name == "architecture":        # one factor at a time around the base network, at n = 10,000
        base = dict(n=10000)
        cfgs = ([dict(depth=d) for d in (2, 4, 6)] + [dict(width=w) for w in (64, 128, 256)] +
                [dict(activation=a) for a in ("softplus", "relu")])
        return [Run("grid_net", noise=z, seed=s, **base, **c) for c in cfgs for z in (0.0, 0.02) for s in range(5)]
    if name == "pointwise":           # representation: pointwise nets, price or vol output, at n = 10,000
        cfgs = [("iv", "softplus"), ("iv", "elu"), ("price", "softplus"), ("price", "elu"), ("price", "relu")]
        return [Run("pointwise", 10000, z, s, target=t, activation=a, **POINTWISE_BUDGET[t])
                for (t, a) in cfgs for z in (0.0, 0.02) for s in range(5)]
    if name == "repairs":             # step 6 (H3): pointwise price networks, softplus, at n = 10,000
        base = dict(n=10000, target="price", activation="softplus", **POINTWISE_BUDGET["price"])
        cfgs = ([dict()] + [dict(deriv_weight=w) for w in REPAIR_WEIGHTS["deriv"]] + [dict(deriv_weight=1.0, deriv_terms="slope")] +
                [dict(penalty_weight=w) for w in REPAIR_WEIGHTS["penalty"]])
        return [Run("pointwise", noise=z, seed=s, **base, **c) for c in cfgs for z in (0.0, 0.02) for s in range(min(seeds, 5))]
    raise ValueError(f"unknown plan {name}")


# Weights of the repairs, fixed from the validation pilot (results/pilot_repairs.json, docs/repairs.md) before the main runs.
REPAIR_WEIGHTS = {"deriv": (0.1, 1.0, 10.0), "penalty": (0.1, 1.0, 10.0, 100.0)}
PLAN_DIRS = {"repairs": os.path.join(data.ROOT, "results", "repairs", "runs")}


# Chosen on the validation set (results/pilot_pointwise.json, docs/experiments.md) with softplus at 5,000 surfaces, by the rule fixed
# beforehand: lowest validation error of the model's own output. The same budget is used for the other activations.
POINTWISE_BUDGET = {"iv": dict(width=128, depth=4, epochs=30, schedule="cosine"),
                    "price": dict(width=64, depth=3, epochs=60, schedule="cosine")}


def run_task(args):
    """Worker entry point for the process pool: one thread per process, errors returned instead of raised."""
    r, out_dir = args
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    try:
        t0 = time.time()
        s = run_one(r, out_dir)
        return r.tag(), s, time.time() - t0, None
    except Exception as e:                                   # keep the pool going; the plan can be re-run to retry
        import traceback
        return r.tag(), None, 0.0, traceback.format_exc()
