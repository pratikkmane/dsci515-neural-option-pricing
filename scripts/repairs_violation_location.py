"""Where do the remaining convexity violations sit (maturity and strike) for the control, the derivative loss and the penalty?
Noise 0, base grid, first 2,000 test surfaces, mean over the seeds present. Needs results/repairs/runs/*.npz (re-run the repairs plan)."""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import data, heston as H, violations as V

te = data.load_test()
res = ~np.isnan(te["iv"][:2000]).reshape(-1, 8, 11)
for name, pat in [("control", "pointwise_price_softplus_w64d3_e60_cosine_n5000_noise0_seed*"), ("derivative loss 1", "*_deriv1_n5000_noise0_seed*"),
                  ("penalty 10", "*_pen10_n5000_noise0_seed*")]:
    by_t, by_k, cnt = np.zeros(8), np.zeros(9), 0
    for f in glob.glob(os.path.join(data.ROOT, "results", "repairs", "runs", pat + ".npz")):
        c = np.where(res, np.load(f)["pred_call"].reshape(-1, 8, 11).astype(float), np.nan)
        fl = np.asarray(V.flags(c, H.STRIKES, H.MATURITIES)["convexity"]).reshape(-1, 8, 9)
        by_t, by_k, cnt = by_t + fl.mean((0, 2)), by_k + fl.mean((0, 1)), cnt + 1
    print(f"{name}: convexity violation rate by maturity {np.round(by_t / cnt, 3)}\n   by interior strike 0.6..1.4 {np.round(by_k / cnt, 3)}")
