"""Pilot for step 6: which weights of the derivative loss and the arbitrage penalty are in a useful range?

Trains pointwise price networks (the step 5 budget, fewer surfaces) with a range of weights and evaluates on the VALIDATION set only
(src/data.py, load_val), never the test set. It fixes the range of weights for the main H3 runs; it does not pick a winner.
Usage: python scripts/pilot_repairs.py [n_surfaces] [workers]
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import data, evaluate, heston as H, violations as V

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = os.path.join(data.ROOT, "results", "pilot_repairs.json")
CONFIGS = ([("sq", 0.0)] + [("deriv", w) for w in (0.01, 0.1, 1.0, 10.0)] + [("slope", w) for w in (0.1, 1.0)] +
           [("penalty", w) for w in (0.01, 0.1, 1.0, 10.0, 100.0)])


def one(args):
    kind, w = args
    import torch
    torch.set_num_threads(1)
    import networks as N
    kw = dict(deriv_weight=w if kind in ("deriv", "slope") else 0.0, deriv_terms="slope" if kind == "slope" else "both",
              penalty_weight=w if kind == "penalty" else 0.0)
    cfg = N.TrainConfig(width=64, depth=3, activation="softplus", epochs=60, batch_size=1024, schedule="cosine", seed=0, **kw)
    tr = data.load_train(N_TRAIN, 0)
    dv = data.load_derivs("train", tr["params"], seed=0)
    t0 = time.time()
    m = N.PointwiseSurrogate(cfg, target="price").fit(tr["params"], tr["call"], tr["iv"], dv)
    val = data.load_val()
    res = evaluate.evaluate_prices(m.predict_call(val["params"]), val)["all"]
    sub = val["params"][:300]
    dense = V.summarise(m.predict_call(sub, H.DENSE_STRIKES, H.DENSE_MATURITIES), H.DENSE_STRIKES, H.DENSE_MATURITIES)
    return dict(kind=kind, weight=w, seconds=time.time() - t0, price_rmse=res["price_rmse"], rmse_iv=res["accuracy"]["rmse_iv"],
                base_surface_rate=res["violations"]["any_violation_surface_rate"],
                dense_cell_rates={k: dense[k]["cell_rate"] for k in ("monotone", "convexity", "calendar")},
                final_terms=m.info["final_terms_price_deriv_penalty"])


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    data.load_train(N_TRAIN, 0); data.load_val()
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(one, CONFIGS))
    for r in rows:
        print(f"{r['kind']:8s} {r['weight']:7g}  price_rmse {r['price_rmse']:.2e}  iv_rmse {r['rmse_iv']:.4f}  base_surf {r['base_surface_rate']:.3f}  "
              f"dense cells {max(r['dense_cell_rates'].values()):.4f}  {r['seconds']:.0f}s")
    json.dump(dict(n_train=N_TRAIN, rows=rows), open(OUT, "w"), indent=2)
