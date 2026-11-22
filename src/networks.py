"""Neural surrogates (step 4): the base-paper grid network and a pointwise network with autodiff strike curvature.

GridSurrogate       the base paper's map from the 5 parameters to the 88-point implied vol surface: ELU layers, RMSE loss on
                    the standardised surface, Adam, batch 32. Unresolved cells (NaN) are masked in the loss, never filled.
PointwiseSurrogate  input (5 scaled parameters, strike, maturity), one output. target="price" outputs the call price (as in the
                    proposal: with ReLU the strike curvature is zero almost everywhere); target="iv" outputs a positive
                    implied vol and the price comes from Black-Scholes, so curvature is differentiated through it. Training
                    rows are the 88 grid points of each surface; evaluation can use any strikes and maturities in the box.
                    Strike derivatives of the price (slope and curvature = implied density) come from autograd.

Everything is float64 and seeded.

Repairs (step 6, pointwise networks, docs/repairs.md). Both start from the squared-error price loss L0 (RMSE) and add one term:
  derivative loss    + deriv_weight * s0 * [RMSE((dC/dK - slope)/s1) + RMSE((d2C/dK2 - density)/s2)], the derivatives taken by
                     autograd at the training points and compared with exact (optionally noisy) strike-derivative labels;
                     deriv_terms="slope" drops the density term. s0, s1, s2 are label scales (price RMS, slope RMS, density RMS
                     per maturity) so that the weights are dimensionless.
  arbitrage penalty  + penalty_weight * s0 * [mean(relu(dC/dK)/s1) + mean(relu(-d2C/dK2)/s2) + mean(relu(-dC/dT)/s0)], a one-sided
                     hinge on the three Carr and Madan conditions at random (strike, maturity) points in the box with the batch's
                     parameters. It uses no labels (the scales above are the only thing it takes from them).
"""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn

import data
import heston as H

DTYPE = torch.float64
K_LO, K_HI = float(H.STRIKES[0]), float(H.STRIKES[-1])
T_LO, T_HI = float(H.MATURITIES[0]), float(H.MATURITIES[-1])
ACTIVATIONS = {"elu": nn.ELU, "softplus": nn.Softplus, "relu": nn.ReLU}
N_CELLS = len(H.MATURITIES) * len(H.STRIKES)


@dataclass
class TrainConfig:
    width: int = 30
    depth: int = 3
    activation: str = "elu"
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0
    schedule: str = "constant"       # "cosine" decays the learning rate to 1 percent of lr over training
    deriv_weight: float = 0.0        # weight of the derivative loss (pointwise networks only)
    deriv_terms: str = "both"        # "both" (slope and density) or "slope"
    penalty_weight: float = 0.0      # weight of the arbitrage penalty (pointwise networks only)


def grid_batch_size(n):
    """Batch size that keeps the number of optimiser steps about the same for every training size: the base paper's 32
    (10,200 surfaces, 319 steps per epoch) up to 10,000 surfaces, then growing in proportion (160 at 50,000, 800 at 250,000)."""
    return 32 if n <= 10_000 else int(round(32 * n / 10_000 / 32) * 32)


def make_scheduler(opt, cfg, steps_per_epoch):
    if cfg.schedule == "constant":
        return None
    if cfg.schedule != "cosine":
        raise ValueError("schedule must be 'constant' or 'cosine'")
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch, eta_min=cfg.lr * 0.01)


def scale_k(k):
    return 2 * (k - K_LO) / (K_HI - K_LO) - 1


def scale_t(t):
    return 2 * (t - T_LO) / (T_HI - T_LO) - 1


def mlp(n_in, n_out, width, depth, activation):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), ACTIVATIONS[activation]()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers).to(DTYPE)


def bs_call(sigma, k, t):
    """Black-Scholes call price with forward 1, in torch (differentiable)."""
    sd = sigma * torch.sqrt(t)
    d1 = -torch.log(k) / sd + 0.5 * sd
    return torch.special.ndtr(d1) - k * torch.special.ndtr(d1 - sd)


def unscale_k(ks):
    return (ks + 1) / 2 * (K_HI - K_LO) + K_LO


def unscale_t(ts):
    return (ts + 1) / 2 * (T_HI - T_LO) + T_LO


def masked_rmse(pred, y, mask):
    return torch.sqrt(((pred - y) ** 2 * mask).sum() / mask.sum())


class GridSurrogate:
    kind = "grid"

    def __init__(self, cfg=None):
        self.cfg = cfg or TrainConfig()

    def fit(self, params, iv):
        c = self.cfg
        torch.manual_seed(c.seed)
        self.std = data.Standardiser().fit(iv)
        z = self.std.transform(iv)
        mask = torch.tensor(~np.isnan(z), dtype=DTYPE)
        y = torch.tensor(np.nan_to_num(z), dtype=DTYPE)
        x = torch.tensor(data.scale_params(params), dtype=DTYPE)
        self.net = mlp(5, N_CELLS, c.width, c.depth, c.activation)
        opt = torch.optim.Adam(self.net.parameters(), lr=c.lr)
        sched = make_scheduler(opt, c, -(-len(x) // c.batch_size))
        self.history = []
        for _ in range(c.epochs):
            perm, total, steps = torch.randperm(len(x)), 0.0, 0
            for i in range(0, len(x), c.batch_size):
                idx = perm[i:i + c.batch_size]
                opt.zero_grad()
                loss = masked_rmse(self.net(x[idx]), y[idx], mask[idx])
                loss.backward()
                opt.step()
                if sched:
                    sched.step()
                total, steps = total + loss.item(), steps + 1
            self.history.append(total / steps)
        self.info = dict(config=asdict(c), final_train_loss_standardised=self.history[-1], n_parameters=sum(p.numel() for p in self.net.parameters()))
        return self

    @torch.no_grad()
    def predict(self, params):
        out = self.net(torch.tensor(data.scale_params(params), dtype=DTYPE)).numpy()
        return self.std.inverse(out)


class PointwiseSurrogate:
    kind = "pointwise"

    def __init__(self, cfg=None, target="price"):
        if target not in ("price", "iv"):
            raise ValueError("target must be 'price' or 'iv'")
        self.cfg, self.target = cfg or TrainConfig(batch_size=1024, epochs=30), target

    # --- model -----------------------------------------------------------------------------------------------------
    def _raw(self, theta, ks, ts):
        return self.net(torch.cat([theta, ks[:, None], ts[:, None]], dim=1))[:, 0]

    def _price(self, theta, ks, ts, k, t):
        out = self._raw(theta, ks, ts)
        if self.target == "price":
            return out
        sigma = nn.functional.softplus(out) * self.scale_iv + 1e-6
        return bs_call(sigma, k, t)

    # --- training --------------------------------------------------------------------------------------------------
    def _price_and_derivs(self, theta, ks_in, ts_in, density=True, calendar=False):
        """Price, dC/dK, d2C/dK2 (if density) and dC/dT (if calendar) at the given inputs by autograd, keeping the graph so that
        a loss on the derivatives can be back-propagated. Derivatives are in strike and maturity units."""
        ks = ks_in.detach().clone().requires_grad_(True)
        ts = ts_in.detach().clone().requires_grad_(calendar)
        price = self._price(theta, ks, ts, unscale_k(ks), unscale_t(ts))
        wrt = [ks] + ([ts] if calendar else [])
        g = torch.autograd.grad(price.sum(), wrt, create_graph=True)
        d1 = g[0] * (2 / (K_HI - K_LO))
        d2 = None
        if density:
            # a network that is piecewise linear in the strike (ReLU) has no dependence of dC/dK on K: the second derivative is 0
            g2 = torch.autograd.grad(g[0].sum(), ks, create_graph=True, allow_unused=True)[0] if g[0].requires_grad else None
            d2 = (torch.zeros_like(ks) if g2 is None else g2) * (2 / (K_HI - K_LO)) ** 2
        dt = g[1] * (2 / (T_HI - T_LO)) if calendar else None
        return price, d1, d2, dt

    def arbitrage_penalty(self, theta, k, t):
        """Mean one-sided violation of the three Carr and Madan conditions at the given (strike, maturity) points, in label scales:
        relu(dC/dK) / s1 + relu(-d2C/dK2) / s2(t) + relu(-dC/dT) / s0. Zero for a surface that satisfies all three there."""
        _, p1, p2, pt = self._price_and_derivs(theta, scale_k(k), scale_t(t), density=True, calendar=True)
        s2 = torch.tensor(np.interp(t.detach().numpy(), H.MATURITIES, self.s2_by_maturity), dtype=DTYPE)
        return (torch.relu(p1) / self.s1).mean() + (torch.relu(-p2) / s2).mean() + (torch.relu(-pt) / self.s0).mean()

    def fit(self, params, call, iv=None, deriv=None):
        """call (n, 88) exact (or noisy) call prices. target='iv' also needs iv (n, 88) with NaN where unresolved.
        deriv = dict(slope (n, 88), density (n, 88)) strike-derivative labels, needed when deriv_weight or penalty_weight > 0
        (the penalty uses them only to set scales)."""
        c = self.cfg
        use_d, use_p = c.deriv_weight > 0, c.penalty_weight > 0
        if (use_d or use_p) and deriv is None:
            raise ValueError("deriv_weight and penalty_weight need strike-derivative labels (for the loss, or for the scales)")
        if (use_d or use_p) and self.target != "price":
            raise ValueError("the repairs are implemented for target='price' (their weights are scaled by the price level)")
        torch.manual_seed(c.seed)
        n = len(params)
        theta = torch.tensor(data.scale_params(params), dtype=DTYPE)
        k_all = torch.tensor(np.tile(H.STRIKES, len(H.MATURITIES)), dtype=DTYPE)       # maturity-major cell layout
        t_all = torch.tensor(np.repeat(H.MATURITIES, len(H.STRIKES)), dtype=DTYPE)
        if self.target == "price":
            y, rows = torch.tensor(call, dtype=DTYPE).reshape(-1), torch.arange(n * N_CELLS)
        else:
            self.scale_iv = float(np.nanmean(iv))
            flat = torch.tensor(iv, dtype=DTYPE).reshape(-1)
            rows, y = torch.nonzero(~torch.isnan(flat))[:, 0], flat
        if use_d or use_p:
            nk = len(H.STRIKES)
            self.s0 = float(np.sqrt(np.mean(np.asarray(call) ** 2)))
            self.s1 = float(np.sqrt(np.mean(np.asarray(deriv["slope"]) ** 2)))
            dens = np.asarray(deriv["density"]).reshape(n, len(H.MATURITIES), nk)
            self.s2_by_maturity = np.sqrt(np.mean(dens ** 2, axis=(0, 2)))
            s2_cell = torch.tensor(np.repeat(self.s2_by_maturity, nk), dtype=DTYPE)
            slope_y = torch.tensor(deriv["slope"], dtype=DTYPE).reshape(-1)
            dens_y = torch.tensor(deriv["density"], dtype=DTYPE).reshape(-1)
        self.net = mlp(7, 1, c.width, c.depth, c.activation)
        opt = torch.optim.Adam(self.net.parameters(), lr=c.lr)
        sched = make_scheduler(opt, c, -(-len(rows) // c.batch_size))
        self.history, self.history_terms = [], []
        for _ in range(c.epochs):
            perm = rows[torch.randperm(len(rows))]
            total, steps, terms = 0.0, 0, np.zeros(3)
            for i in range(0, len(perm), c.batch_size):
                r = perm[i:i + c.batch_size]
                s, cell = r // N_CELLS, r % N_CELLS
                k, t = k_all[cell], t_all[cell]
                opt.zero_grad()
                if use_d:
                    price, d1, d2, _ = self._price_and_derivs(theta[s], scale_k(k), scale_t(t), density=c.deriv_terms == "both")
                    l0 = torch.sqrt(torch.mean((price - y[r]) ** 2))
                    ld = torch.sqrt(torch.mean(((d1 - slope_y[r]) / self.s1) ** 2))
                    if c.deriv_terms == "both":
                        ld = ld + torch.sqrt(torch.mean(((d2 - dens_y[r]) / s2_cell[cell]) ** 2))
                    loss = l0 + c.deriv_weight * self.s0 * ld
                else:
                    out = self._raw(theta[s], scale_k(k), scale_t(t))
                    pred = out if self.target == "price" else nn.functional.softplus(out) * self.scale_iv + 1e-6
                    l0 = torch.sqrt(torch.mean((pred - y[r]) ** 2))
                    ld, loss = torch.zeros(()), l0
                lp = torch.zeros(())
                if use_p:
                    kr = K_LO + (K_HI - K_LO) * torch.rand(len(r), dtype=DTYPE)
                    tr = T_LO + (T_HI - T_LO) * torch.rand(len(r), dtype=DTYPE)
                    lp = self.arbitrage_penalty(theta[s], kr, tr)
                    loss = loss + c.penalty_weight * self.s0 * lp
                loss.backward()
                opt.step()
                if sched:
                    sched.step()
                total, steps = total + loss.item(), steps + 1
                terms += [l0.item(), ld.item(), lp.item()]
            self.history.append(total / steps)
            self.history_terms.append((terms / steps).tolist())
        self.info = dict(config=asdict(c), target=self.target, final_train_loss=self.history[-1],
                         final_terms_price_deriv_penalty=self.history_terms[-1],
                         n_parameters=sum(p.numel() for p in self.net.parameters()))
        return self

    # --- prediction on any grid ------------------------------------------------------------------------------------
    def _grid_inputs(self, params, strikes, maturities):
        n, nt, nk = len(params), len(maturities), len(strikes)
        theta = torch.tensor(np.repeat(data.scale_params(params), nt * nk, axis=0), dtype=DTYPE)
        k = torch.tensor(np.tile(np.asarray(strikes, float), n * nt), dtype=DTYPE)
        t = torch.tensor(np.tile(np.repeat(np.asarray(maturities, float), nk), n), dtype=DTYPE)
        return theta, k, t

    def _chunks(self, n, strikes, maturities, max_rows=400_000):
        per = max(1, max_rows // (len(strikes) * len(maturities)))
        return [slice(i, min(i + per, n)) for i in range(0, n, per)]

    @torch.no_grad()
    def predict_call(self, params, strikes=H.STRIKES, maturities=H.MATURITIES):
        """Call prices (n, T, K) on any strikes and maturities inside the training box."""
        params = np.atleast_2d(params)
        out = []
        for sl in self._chunks(len(params), strikes, maturities):
            theta, k, t = self._grid_inputs(params[sl], strikes, maturities)
            out.append(self._price(theta, scale_k(k), scale_t(t), k, t).numpy().reshape(-1, len(maturities), len(strikes)))
        return np.concatenate(out)

    def strike_derivatives(self, params, strikes=H.STRIKES, maturities=H.MATURITIES):
        """Price, dC/dK and d2C/dK2 (the implied density) by autograd, each (n, T, K)."""
        params = np.atleast_2d(params)
        res = [[], [], []]
        # the double-backward graph is much larger than a forward pass, so use small chunks (about 0.5 GB for a 128 x 4 net)
        for sl in self._chunks(len(params), strikes, maturities, max_rows=25_000):
            theta, k, t = self._grid_inputs(params[sl], strikes, maturities)
            ks = scale_k(k).clone().requires_grad_(True)
            kk = (ks + 1) / 2 * (K_HI - K_LO) + K_LO                  # strike as a function of the input being differentiated
            price = self._price(theta, ks, scale_t(t), kk, t)
            d1 = torch.autograd.grad(price.sum(), ks, create_graph=True)[0]
            d2 = torch.autograd.grad(d1.sum(), ks)[0]
            scale = 2 / (K_HI - K_LO)                                  # d/dK = scale * d/dks
            shape = (-1, len(maturities), len(strikes))
            for store, v in zip(res, (price.detach(), d1.detach() * scale, d2.detach() * scale ** 2)):
                store.append(v.numpy().reshape(shape))
        return tuple(np.concatenate(r) for r in res)
