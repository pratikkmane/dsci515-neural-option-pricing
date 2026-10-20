"""Where does the reproduction's worst-case error come from?

Loads the weights saved by src/reproduce_heston.py, rebuilds the identical 85/15 split (random_state 42) and
reports relative implied-vol error per grid cell, per parameter bin, and for the worst surfaces.
Writes results/reproduction_error_analysis.json and results/reproduction_error_cells.png.
"""
import argparse, json, os, sys
import pickle, zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reproduce_heston import LB, UB, STRIKES, MATURITIES, load

NAMES = ["xi0", "rho", "sigma", "theta", "kappa"]

def load_state_dict(path):
    """Read a PyTorch .pt state dict into numpy arrays without importing torch (float64 weights, CPU)."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    class Storage:
        def __init__(self, key): self.key = key
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(st, off, size, stride, *_):
                    buf = np.frombuffer(z.read(f"{root}/data/{st.key}"), dtype=np.float64)
                    return np.lib.stride_tricks.as_strided(buf[off:], shape=size,
                                                           strides=[8*t for t in stride]).copy()
                return rebuild
            if name == "OrderedDict":
                import collections; return collections.OrderedDict
            if name == "DoubleStorage":
                return float
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            return Storage(pid[2])
    return U(z.open(f"{root}/data.pkl")).load()

def forward(sd, x):
    """5-30-30-30-88 ELU network, as in src/reproduce_heston.py."""
    elu = lambda v: np.where(v > 0, v, np.expm1(np.minimum(v, 0)))
    h = x
    for i in (0, 2, 4):
        h = elu(h @ sd[f"{i}.weight"].T + sd[f"{i}.bias"])
    return h @ sd["6.weight"].T + sd["6.bias"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/HestonTrainSet.txt.gz")
    ap.add_argument("--weights", default="results/reproduction_weights.pt")
    ap.add_argument("--out", default="results/reproduction_error_analysis.json")
    a = ap.parse_args()

    params, iv = load(a.data)
    P_tr, P_te, S_tr, S_te = train_test_split(params, iv, test_size=0.15, random_state=42)
    sc = StandardScaler().fit(S_tr)
    Xte = 2*(P_te-(UB+LB)/2)/(UB-LB)
    pred = sc.inverse_transform(forward(load_state_dict(a.weights), Xte))
    rel = 100*np.abs((S_te-pred)/S_te)               # (n_test, 88), maturity-major as in the notebook
    cell = lambda v: v.reshape(len(MATURITIES), len(STRIKES))

    res = dict(overall_mean_pct=float(rel.mean()), overall_median_pct=float(np.median(rel)),
               overall_p99_pct=float(np.quantile(rel, 0.99)), overall_max_pct=float(rel.max()),
               share_cells_above_5pct=float((rel > 5).mean()), share_cells_above_1pct=float((rel > 1).mean()))
    res["cell_mean_pct"] = cell(rel.mean(0)).round(3).tolist()
    res["cell_max_pct"] = cell(rel.max(0)).round(3).tolist()
    res["strikes"] = STRIKES.tolist(); res["maturities"] = MATURITIES.tolist()

    surf_max = rel.max(1)
    worst = np.argsort(-surf_max)[:10]
    res["worst_surfaces"] = [dict(test_index=int(i), max_rel_pct=float(surf_max[i]),
                                  worst_cell=dict(maturity=float(MATURITIES[rel[i].argmax()//len(STRIKES)]),
                                                  strike=float(STRIKES[rel[i].argmax() % len(STRIKES)])),
                                  params=dict(zip(NAMES, map(float, P_te[i])))) for i in worst]

    # Does error depend on the parameters? Mean per-surface error by quintile of each parameter.
    surf_mean = rel.mean(1)
    byp = {}
    for j, n in enumerate(NAMES):
        edges = np.quantile(P_te[:, j], np.linspace(0, 1, 6))
        q = np.clip(np.searchsorted(edges, P_te[:, j], side="right")-1, 0, 4)
        byp[n] = dict(quintile_mean_pct=[float(surf_mean[q == k].mean()) for k in range(5)],
                      quintile_max_pct=[float(surf_max[q == k].max()) for k in range(5)])
    res["by_parameter_quintile"] = byp
    res["spearman_surface_mean_error_vs_param"] = {
        n: float(__import__("scipy.stats").stats.spearmanr(P_te[:, j], surf_mean)[0]) for j, n in enumerate(NAMES)}
    # Is the worst case a low-IV artefact? relative error is large when the true vol is small.
    res["spearman_cell_rel_err_vs_true_iv"] = float(__import__("scipy.stats").stats.spearmanr(S_te.ravel(), rel.ravel())[0])
    res["true_iv_at_worst_cell"] = float(S_te.ravel()[rel.argmax()])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for axi, (m, t) in zip(ax, [(cell(rel.mean(0)), "Mean relative error (%)"), (cell(rel.max(0)), "Max relative error (%)")]):
        im = axi.imshow(m); axi.set_title(t); plt.colorbar(im, ax=axi, format="%.1f")
        axi.set_xticks(range(len(STRIKES))); axi.set_xticklabels(STRIKES, fontsize=7)
        axi.set_yticks(range(len(MATURITIES))); axi.set_yticklabels(MATURITIES, fontsize=7)
        axi.set_xlabel("Strike"); axi.set_ylabel("Maturity")
    plt.tight_layout(); plt.savefig(a.out.replace("analysis.json", "cells.png"), dpi=150)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, indent=2))

if __name__ == "__main__":
    main()
