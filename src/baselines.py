"""Classical baselines for the 88-point implied vol surface (step 4): do other flexible regressors also break the rules?

All three map the five Heston parameters (scaled to [-1, 1]) to the surface and share the same preparation:
  1. standardise each of the 88 columns using valid (non-NaN) entries only,
  2. fill the missing cells by an iterated low-rank (PCA) reconstruction, for FITTING ONLY (evaluation never uses filled
     values: unresolved cells are masked, see src/evaluate.py),
  3. (gradient boosting and Gaussian process only) project onto the leading principal components and fit one model per score.

RidgePoly           polynomial features of the parameters, degree and penalty chosen by K-fold cross-validation
PCAGradientBoosting HistGradientBoostingRegressor per principal component score. The number of components is chosen so that
                    the linear PCA truncation error is at most `target_rmse` (5e-4 in implied vol); on this data that
                    takes about 30 components, and the achieved truncation error is stored in `info`.
PCAGaussianProcess  Gaussian process per score with an ARD RBF kernel. Exact GPs cost O(n^3), so the number of training
                    points is capped (`max_train`, default 3000, a random subset) and the kernel is fitted on a smaller
                    subset (`n_hyper`, default 800, one restart); the cap is recorded in `info` and must be quoted with
                    any GP result. Component count is chosen as for gradient boosting.

Each model has fit(params, iv) -> self, predict(params) -> iv (n, 88) and an `info` dict of what was chosen.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

import data


def complete_low_rank(z, rank=10, iters=30):
    """Fill NaN entries of a (standardised) matrix with an iterated rank-`rank` reconstruction. Observed entries are kept."""
    miss = np.isnan(z)
    if not miss.any():
        return z.copy()
    x = np.where(miss, 0.0, z)
    for _ in range(iters):
        mu = x.mean(0)
        c = x - mu
        _, v = np.linalg.eigh(c.T @ c / len(x))
        v = v[:, -rank:]
        x[miss] = (mu + (c @ v) @ v.T)[miss]
    return x


class _Surface:
    """Standardise, fill (for fitting only) and optionally project onto principal components."""

    def __init__(self, rank=10):
        self.rank = rank

    def fit(self, iv):
        self.std = data.Standardiser().fit(iv)
        z = complete_low_rank(self.std.transform(iv), self.rank)
        self.mu = z.mean(0)
        c = z - self.mu
        w, v = np.linalg.eigh(c.T @ c / len(z))
        order = np.argsort(w)[::-1]
        self.components = v[:, order]           # (88, 88), strongest first
        self.explained = (w[order] / w.sum())
        self.z = z
        return self

    def scores(self, k):
        return (self.z - self.mu) @ self.components[:, :k]

    def from_scores(self, s):
        k = s.shape[1]
        return self.std.inverse(s @ self.components[:, :k].T + self.mu)

    def choose_k(self, iv, target_rmse=5e-4, k_max=40):
        """Smallest number of components whose truncation error (implied vol RMSE on observed cells) is at most the target."""
        ok = ~np.isnan(iv)
        s_all = (self.z - self.mu) @ self.components[:, :k_max]
        for k in range(1, k_max + 1):
            rec = self.std.inverse(s_all[:, :k] @ self.components[:, :k].T + self.mu)
            if np.sqrt(np.mean((rec - iv)[ok] ** 2)) <= target_rmse:
                return k
        return k_max

    def reconstruction_rmse(self, iv, k):
        """RMSE in implied vol units of reconstructing the observed cells from k components (the truncation floor)."""
        rec = self.from_scores(self.scores(k))
        ok = ~np.isnan(iv)
        return float(np.sqrt(np.mean((rec - iv)[ok] ** 2)))


class RidgePoly:
    name = "ridge_poly"

    def __init__(self, degrees=(2, 3, 4, 5, 6, 7, 8), alphas=(1e-10, 1e-8, 1e-6, 1e-4), cv=3, rank=10, max_val_rows=20000,
                 cv_max_rows=40000, max_features=800, seed=0):
        self.degrees, self.alphas, self.cv, self.rank, self.max_val_rows, self.seed = degrees, alphas, cv, rank, max_val_rows, seed
        self.cv_max_rows, self.max_features = cv_max_rows, max_features

    def fit(self, params, iv):
        x = data.scale_params(params)
        self.surf = _Surface(self.rank).fit(iv)
        z = self.surf.z
        rng = np.random.default_rng(self.seed)
        cv_mse = {}
        # cross-validation runs on at most cv_max_rows rows (cost), and degrees with more than max_features features are
        # skipped (memory); the chosen degree and penalty are then refitted on all rows
        sel = rng.permutation(len(x))[:self.cv_max_rows]
        x_cv, z_cv = x[sel], z[sel]
        for d in self.degrees:
            poly = PolynomialFeatures(d, include_bias=False)
            if poly.fit(x_cv[:2]).n_output_features_ > self.max_features:
                continue
            phi = poly.fit_transform(x_cv)
            if phi.shape[1] > 0.5 * len(x_cv) * (1 - 1 / self.cv):    # more features than half the fitting rows: skip
                continue
            errs = {a: [] for a in self.alphas}
            for tr, va in KFold(self.cv, shuffle=True, random_state=self.seed).split(phi):
                va = rng.choice(va, min(len(va), self.max_val_rows), replace=False)
                m, s = phi[tr].mean(0), phi[tr].std(0) + 1e-12
                a_mat = (phi[tr] - m) / s
                ym = z_cv[tr].mean(0)
                w, v = np.linalg.eigh(a_mat.T @ a_mat)
                aty = v.T @ (a_mat.T @ (z_cv[tr] - ym))
                av = ((phi[va] - m) / s) @ v
                for a in self.alphas:
                    pred = av @ (aty / (w + a * len(tr))[:, None]) + ym
                    errs[a].append(np.mean((z_cv[va] - pred) ** 2))
            for a in self.alphas:
                cv_mse[(d, a)] = float(np.mean(errs[a]))
        if not cv_mse:
            raise ValueError(f"too few training rows ({len(x)}) for the requested polynomial degrees {self.degrees}")
        (d, a), best = min(cv_mse.items(), key=lambda kv: kv[1])
        # Final fit on all rows by accumulating the normal equations in blocks, so memory does not grow with n
        # (a 250,000 x 792 feature matrix, built twice, needs about 5 GB).
        self.poly = PolynomialFeatures(d, include_bias=False).fit(x[:2])
        blocks = [slice(i, min(i + 20000, len(x))) for i in range(0, len(x), 20000)]
        sums = sq = 0.0
        for b in blocks:
            f = self.poly.transform(x[b])
            sums, sq = sums + f.sum(0), sq + (f ** 2).sum(0)
        self.m = sums / len(x)
        self.s = np.sqrt(np.maximum(sq / len(x) - self.m ** 2, 0.0)) + 1e-12
        self.ym = z.mean(0)
        n_feat = self.poly.n_output_features_
        gram, rhs = np.zeros((n_feat, n_feat)), np.zeros((n_feat, z.shape[1]))
        for b in blocks:
            a_blk = (self.poly.transform(x[b]) - self.m) / self.s
            gram += a_blk.T @ a_blk
            rhs += a_blk.T @ (z[b] - self.ym)
        self.coef = np.linalg.solve(gram + a * len(x) * np.eye(n_feat), rhs)
        self.info = dict(degree=d, alpha=a, cv_mse_standardised=best, n_features=n_feat,
                         cv_folds=self.cv, cv_rows=int(len(x_cv)), degrees_tried=[dd for dd in self.degrees if any(k[0] == dd for k in cv_mse)])
        return self

    def predict(self, params):
        phi = (self.poly.transform(data.scale_params(params)) - self.m) / self.s
        return self.surf.std.inverse(phi @ self.coef + self.ym)


class PCAGradientBoosting:
    name = "gradient_boosting"

    def __init__(self, n_components=None, target_rmse=5e-4, k_max=40, max_iter=500, learning_rate=0.1, max_leaf_nodes=31, seed=0):
        self.k_arg, self.target, self.k_max = n_components, target_rmse, k_max
        self.max_iter, self.lr, self.leaves, self.seed = max_iter, learning_rate, max_leaf_nodes, seed

    def fit(self, params, iv):
        x = data.scale_params(params)
        self.surf = _Surface(10).fit(iv)
        self.k = self.k_arg or self.surf.choose_k(iv, self.target, self.k_max)
        s = self.surf.scores(self.k)
        self.models = [HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=self.lr, max_leaf_nodes=self.leaves,
                                                     early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                                     random_state=self.seed).fit(x, s[:, i]) for i in range(self.k)]
        self.info = dict(n_components=int(self.k), explained_variance=float(self.surf.explained[:self.k].sum()),
                         pca_truncation_rmse_iv=self.surf.reconstruction_rmse(iv, self.k),
                         iterations=[int(m.n_iter_) for m in self.models])
        return self

    def predict(self, params):
        x = data.scale_params(params)
        return self.surf.from_scores(np.column_stack([m.predict(x) for m in self.models]))


class PCAGaussianProcess:
    name = "gaussian_process"

    def __init__(self, n_components=None, target_rmse=5e-4, k_max=40, max_train=3000, n_hyper=800, restarts=1, seed=0):
        self.k_arg, self.target, self.k_max = n_components, target_rmse, k_max
        self.max_train, self.n_hyper, self.restarts, self.seed = max_train, n_hyper, restarts, seed

    def fit(self, params, iv):
        x = data.scale_params(params)
        self.surf = _Surface(10).fit(iv)
        self.k = self.k_arg or self.surf.choose_k(iv, self.target, self.k_max)
        s = self.surf.scores(self.k)
        rng = np.random.default_rng(self.seed)
        use = rng.permutation(len(x))[:self.max_train]
        hyper = use[:self.n_hyper]
        self.models = []
        for i in range(self.k):
            # length scales are bounded below by 0.2 (inputs span [-1, 1]); without it the optimiser can collapse to a
            # tiny scale that fits noise (seen on the 3rd and 4th components)
            kern = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(5), (0.2, 1e2)) + WhiteKernel(1e-6, (1e-12, 1e-1))
            first = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=self.restarts,
                                             random_state=self.seed).fit(x[hyper], s[hyper, i])
            self.models.append(GaussianProcessRegressor(first.kernel_, optimizer=None, normalize_y=True).fit(x[use], s[use, i]))
        self.info = dict(n_components=self.k, n_train_used=int(len(use)), n_train_available=int(len(x)),
                         n_for_kernel_fit=int(len(hyper)), capped=bool(len(x) > self.max_train),
                         explained_variance=float(self.surf.explained[:self.k].sum()),
                         pca_truncation_rmse_iv=self.surf.reconstruction_rmse(iv, self.k))
        return self

    def predict(self, params):
        x = data.scale_params(params)
        return self.surf.from_scores(np.column_stack([m.predict(x) for m in self.models]))


MODELS = {c.name: c for c in (RidgePoly, PCAGradientBoosting, PCAGaussianProcess)}
