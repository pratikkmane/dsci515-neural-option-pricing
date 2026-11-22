"""Exact Heston datasets for the experiments (step 4).

Training sets are the first n rows of a scrambled Sobol sequence, so a smaller set is a prefix of a larger one for the same
seed (clean learning curves). Different seeds give different scrambles, hence different training sets. The test set is
fixed (its own seed) and labelled interior or boundary: 'interior' means every parameter lies in the central 80 percent of
its range, 'boundary' means at least one lies in the outer 10 percent at either end.

Targets are implied vols on the 88-point base grid (maturity-major) from exact prices. Cells whose out-of-the-money price is
below heston.PRICE_FLOOR have no implied vol and are NaN: they are masked, never filled (CLAUDE.md, section 6). Everything is
cached under data/generated/, which git ignores.
"""
import os
import numpy as np
import heston as H

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GEN_DIR = os.path.join(ROOT, "data", "generated")
TRAIN_SEED_BASE = 1000     # training seed s uses Sobol scramble 1000 + s
TEST_SEED = 2002
VAL_SEED = 3003        # validation set: used only to choose training budgets, never for reported results
N_TEST = 20_000
NOISE_LEVELS = (0.0, 0.001, 0.005, 0.02)      # multiplicative, as in the proposal (0, 0.1, 0.5, 2 percent)
GRID_SHAPE = (len(H.MATURITIES), len(H.STRIKES))


def scale_params(p):
    """Parameters to [-1, 1], as in the base paper."""
    return 2 * (np.asarray(p, float) - (H.UB + H.LB) / 2) / (H.UB - H.LB)


def region(params, band=0.10):
    """'interior' (all parameters within the central 80 percent of their range) or 'boundary', as a boolean array."""
    u = (np.asarray(params, float) - H.LB) / (H.UB - H.LB)
    return np.where(((u >= band) & (u <= 1 - band)).all(axis=1), "interior", "boundary")


def surfaces(params):
    """Exact implied vols (n, 88, NaN where unresolved) and call prices (n, 88) on the base grid, maturity-major."""
    params = np.atleast_2d(params)
    otm = H.otm_prices(params)
    k, t = H.STRIKES[None, None, :], H.MATURITIES[None, :, None]
    iv = H.implied_vol_otm(otm, k, t)
    call = np.where(k >= 1, otm, otm + 1 - k)
    return iv.reshape(len(params), -1), call.reshape(len(params), -1)


def _make(params, workers):
    if workers <= 1 or len(params) < 2000:
        return surfaces(params)
    from concurrent.futures import ProcessPoolExecutor
    chunks = np.array_split(params, workers * 4)
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(surfaces, chunks))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _cached(name, params_fn, workers):
    path = os.path.join(GEN_DIR, name + ".npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    params = params_fn()
    iv, call = _make(params, workers)
    out = dict(params=params, iv=iv, call=call)
    os.makedirs(GEN_DIR, exist_ok=True)
    np.savez(path, **out)
    return out


def load_train(n, seed=0, workers=1):
    """Training set: dict(params (n, 5), iv (n, 88), call (n, 88)).

    If a larger set for the same seed is already cached, the first n rows are used (training sets are nested)."""
    import glob, re
    sizes = sorted(int(m.group(1)) for f in glob.glob(os.path.join(GEN_DIR, f"train_n*_seed{seed}.npz"))
                   if (m := re.search(rf"train_n(\d+)_seed{seed}\.npz$", f)) and int(m.group(1)) >= n)
    if sizes:
        d = np.load(os.path.join(GEN_DIR, f"train_n{sizes[0]}_seed{seed}.npz"))
        return {k: d[k][:n] for k in d.files}
    return _cached(f"train_n{n}_seed{seed}", lambda: H.sobol_params(n, seed=TRAIN_SEED_BASE + seed), workers)


def load_val(workers=1, n=2000):
    """Validation set (own seed) for choosing training budgets: dict(params, iv, call, region)."""
    d = _cached(f"val_n{n}_seed{VAL_SEED}", lambda: H.sobol_params(n, seed=VAL_SEED), workers)
    d["region"] = region(d["params"])
    return d


def load_test(workers=1, n=N_TEST):
    """Fixed test set plus its interior/boundary label: dict(params, iv, call, region)."""
    d = _cached(f"test_n{n}_seed{TEST_SEED}", lambda: H.sobol_params(n, seed=TEST_SEED), workers)
    d["region"] = region(d["params"])
    return d


def add_noise(y, level, seed):
    """Multiplicative Gaussian noise y * (1 + level * N(0, 1)); NaN stays NaN. level 0 returns y unchanged."""
    y = np.asarray(y, float)
    if level == 0:
        return y.copy()
    return y * (1 + level * np.random.default_rng(seed).standard_normal(y.shape))


class Standardiser:
    """Per-column mean and standard deviation of an (n, d) array that may contain NaN (fitted on valid entries only)."""

    def fit(self, y):
        valid = ~np.isnan(y)
        count = valid.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nansum(y, 0) / count
            var = np.nansum((y - mean) ** 2, 0) / count
        self.mean = np.where(count > 0, mean, 0.0)
        std = np.sqrt(np.where(count > 1, var, 1.0))
        self.std = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean
