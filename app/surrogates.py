"""Numpy-only evaluation of the trained demo models (no torch needed in the deployed app). Weights come from scripts/train_demo_models.py."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def _act(name):
    if name == "elu":
        return lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    if name == "softplus":
        return lambda x: np.logaddexp(0.0, x)
    raise ValueError(name)


def _mlp(w, x):
    act, i = _act(str(w["activation"])), 0
    while f"W{i + 1}" in w:
        x = act(x @ w[f"W{i}"].T + w[f"b{i}"])
        i += 1
    return x @ w[f"W{i}"].T + w[f"b{i}"]


def scale_params(p, lb, ub):
    return 2 * (np.asarray(p, float) - (ub + lb) / 2) / (ub - lb)


class Grid:
    """Base-paper network: parameters -> 88 implied vols on the base grid."""
    kind = "grid"

    def __init__(self, path, lb, ub):
        self.w, self.lb, self.ub = dict(np.load(path)), lb, ub

    def predict_iv(self, params):
        z = _mlp(self.w, scale_params(np.atleast_2d(params), self.lb, self.ub))
        return z * self.w["std"] + self.w["mean"]


class Ridge:
    kind = "grid"

    def __init__(self, path, lb, ub):
        from sklearn.preprocessing import PolynomialFeatures
        self.w, self.lb, self.ub = dict(np.load(path)), lb, ub
        self.poly = PolynomialFeatures(int(self.w["degree"]), include_bias=False).fit(np.zeros((2, 5)))

    def predict_iv(self, params):
        phi = (self.poly.transform(scale_params(np.atleast_2d(params), self.lb, self.ub)) - self.w["m"]) / self.w["s"]
        return (phi @ self.w["coef"] + self.w["ym"]) * self.w["std"] + self.w["mean"]


class Pointwise:
    """(parameters, strike, maturity) -> call price, on any grid inside the training box."""
    kind = "pointwise"

    def __init__(self, path, lb, ub, strikes, maturities):
        self.w, self.lb, self.ub = dict(np.load(path)), lb, ub
        self.k_lo, self.k_hi, self.t_lo, self.t_hi = strikes[0], strikes[-1], maturities[0], maturities[-1]

    def predict_call(self, params, strikes, maturities):
        """(len(maturities), len(strikes)) call prices for one parameter set."""
        strikes, maturities = np.asarray(strikes, float), np.asarray(maturities, float)
        k = np.tile(strikes, len(maturities))
        t = np.repeat(maturities, len(strikes))
        theta = np.repeat(scale_params(params, self.lb, self.ub).reshape(1, 5), len(k), axis=0)
        ks, ts = 2 * (k - self.k_lo) / (self.k_hi - self.k_lo) - 1, 2 * (t - self.t_lo) / (self.t_hi - self.t_lo) - 1
        out = _mlp(self.w, np.column_stack([theta, ks, ts]))[:, 0]
        return out.reshape(len(maturities), len(strikes))


def load_all(lb, ub, strikes, maturities):
    """Models whose weight files exist, as an ordered dict name -> (label, object)."""
    spec = [("grid", "Base-paper network (grid output)", lambda p: Grid(p, lb, ub)),
            ("ridge", "Polynomial ridge", lambda p: Ridge(p, lb, ub)),
            ("price", "Price network, squared error", lambda p: Pointwise(p, lb, ub, strikes, maturities)),
            ("price_pen", "Price network + arbitrage penalty", lambda p: Pointwise(p, lb, ub, strikes, maturities))]
    out = {}
    for name, label, make in spec:
        path = os.path.join(MODEL_DIR, name + ".npz")
        if os.path.exists(path):
            out[name] = (label, make(path))
    return out
