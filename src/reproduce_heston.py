"""PyTorch port of the Heston surrogate in amuguruza/NN-StochVol-Calibrations (Heston/NNHeston.ipynb).

Same data file, 85/15 split (random_state=42), parameter scaling to [-1, 1], 5-30-30-30-88 ELU net,
RMSE loss, Adam, batch 32. Maps Heston parameters -> 88 implied vols (11 strikes x 8 maturities).
In the original notebook the surface is standardised with StandardScaler and parameters with the bounds below.
"""
import argparse, gzip, json, time
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LB = np.array([0.0001, -0.95, 0.01, 0.01, 1.0])
UB = np.array([0.04, -0.1, 1.0, 0.2, 10.0])
STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])

def load(path):
    with gzip.GzipFile(path, "r") as f:
        dat = np.load(f)
    return dat[:, :5], dat[:, 5:]   # surfaces? see main(): columns 0-4 are params, 5.. are IVs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results/reproduction.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_default_dtype(torch.float64)

    params, iv = load(a.data)
    print("data: params", params.shape, "iv", iv.shape)
    # notebook: xx=dat[:,:5] (named xx), yy=dat[:,5:]; train_test_split(yy, xx) then x=surface(88), y=params(5)
    P_tr, P_te, S_tr, S_te = train_test_split(params, iv, test_size=0.15, random_state=42)
    # the notebook's naming is swapped; here P = parameters, S = surfaces. Check shapes below.
    print("P_tr", P_tr.shape, "S_tr", S_tr.shape)
    sc = StandardScaler().fit(S_tr)
    Xtr = torch.tensor((2*(P_tr-(UB+LB)/2)/(UB-LB))); Xte = torch.tensor((2*(P_te-(UB+LB)/2)/(UB-LB)))
    Ytr = torch.tensor(sc.transform(S_tr)); Yte = torch.tensor(sc.transform(S_te))

    net = torch.nn.Sequential(torch.nn.Linear(5,30), torch.nn.ELU(), torch.nn.Linear(30,30), torch.nn.ELU(),
                              torch.nn.Linear(30,30), torch.nn.ELU(), torch.nn.Linear(30,88))
    opt = torch.optim.Adam(net.parameters())
    rmse = lambda p, y: torch.sqrt(torch.mean((p-y)**2))
    hist = []; t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            opt.zero_grad(); loss = rmse(net(Xtr[idx]), Ytr[idx]); loss.backward(); opt.step()
        with torch.no_grad():
            va = rmse(net(Xte), Yte).item()
        hist.append(va)
        if ep % 10 == 0 or ep == a.epochs-1:
            print(f"epoch {ep+1}/{a.epochs} val_rmse(std units)={va:.5f} elapsed={time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        pred = sc.inverse_transform(net(Xte).numpy())
    rel = 100*np.abs((S_te-pred)/S_te)
    res = dict(n_train=len(Xtr), n_test=len(Xte), epochs=a.epochs,
               mean_rel_err_pct=float(rel.mean()), max_rel_err_pct=float(rel.max()),
               rmse_iv_points=float(np.sqrt(np.mean((S_te-pred)**2))),
               val_rmse_std_units_final=hist[-1], seconds=time.time()-t0)
    print(json.dumps(res, indent=2))
    json.dump(res, open(a.out, "w"), indent=2)
    torch.save(net.state_dict(), a.out.replace(".json", "_weights.pt"))

if __name__ == "__main__":
    main()
