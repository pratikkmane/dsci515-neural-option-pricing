"""Neural surrogates (step 4): the base-paper grid network and a pointwise network with autodiff strike curvature.

GridSurrogate       the base paper's map from the 5 parameters to the 88-point implied vol surface: ELU layers, RMSE loss on
                    the standardised surface, Adam, batch 32. Unresolved cells (NaN) are masked in the loss, never filled.
PointwiseSurrogate  input (5 scaled parameters, strike, maturity), one output. target="price" outputs the call price (as in the
                    proposal: with ReLU the strike curvature is zero almost everywhere); target="iv" outputs a positive
                    implied vol and the price comes from Black-Scholes, so curvature is differentiated through it. Training
                    rows are the 88 grid points of each surface; evaluation can use any strikes and maturities in the box.
                    Strike derivatives of the price (slope and curvature = implied density) come from autograd.

Everything is float64 and seeded. The derivative loss and the arbitrage penalty (step 6) are not here yet.
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
    def fit(self, params, call, iv=None):
        """call (n, 88) exact (or noisy) call prices. target='iv' also needs iv (n, 88) with NaN where unresolved."""
        c = self.cfg
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
        self.net = mlp(7, 1, c.width, c.depth, c.activation)
        opt = torch.optim.Adam(self.net.parameters(), lr=c.lr)
        sched = make_scheduler(opt, c, -(-len(rows) // c.batch_size))
        self.history = []
        for _ in range(c.epochs):
            perm = rows[torch.randperm(len(rows))]
            total, steps = 0.0, 0
            for i in range(0, len(perm), c.batch_size):
                r = perm[i:i + c.batch_size]
                s, cell = r // N_CELLS, r % N_CELLS
                k, t = k_all[cell], t_all[cell]
                opt.zero_grad()
                out = self._raw(theta[s], scale_k(k), scale_t(t))
                pred = out if self.target == "price" else nn.functional.softplus(out) * self.scale_iv + 1e-6
                loss = torch.sqrt(torch.mean((pred - y[r]) ** 2))
                loss.backward()
                opt.step()
                if sched:
                    sched.step()
                total, steps = total + loss.item(), steps + 1
            self.history.append(total / steps)
        self.info = dict(config=asdict(c), target=self.target, final_train_loss=self.history[-1],
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
        for sl in self._chunks(len(params), strikes, maturities):
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
