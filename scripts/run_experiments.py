"""Run a named plan of experiments in parallel (step 5). Resumable: runs that already have a result are skipped.

Examples:
  python scripts/run_experiments.py core_small --processes 4
  python scripts/run_experiments.py pointwise --processes 4 --seeds 5
  python scripts/run_experiments.py core_small --dry-run

Plans are defined in src/experiment.py. Results go to results/runs/ (JSON committed, NPZ not).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")                            # one thread per worker: these models are small
import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data, experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N missing runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None, help="default: results/runs, or the plan's own folder (experiment.PLAN_DIRS)")
    a = ap.parse_args()

    a.out_dir = a.out_dir or experiment.PLAN_DIRS.get(a.plan, experiment.RUN_DIR)
    runs = experiment.plan(a.plan, a.seeds)
    todo = [r for r in runs if not os.path.exists(os.path.join(a.out_dir, r.tag() + ".json"))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"plan {a.plan}: {len(runs)} runs, {len(runs) - len([r for r in runs if r in todo]) if not a.limit else '?'} done, {len(todo)} to run", flush=True)
    if a.dry_run:
        for r in todo[:20]:
            print("  ", r.tag())
        return

    # generate each seed's largest training set once, before the workers start (so they only read the cache)
    nmax = {}
    for r in todo:
        nmax[r.seed] = max(nmax.get(r.seed, 0), r.n)
    t0 = time.time()
    data.load_test(workers=a.processes)
    for seed, n in sorted(nmax.items()):
        data.load_train(n, seed, workers=a.processes)
    print(f"data ready in {time.time() - t0:.0f}s", flush=True)

    done, failed, t0 = 0, [], time.time()
    with ProcessPoolExecutor(a.processes, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(experiment.run_task, (r, a.out_dir)) for r in todo]
        for f in as_completed(futs):
            try:
                tag, s, secs, err = f.result()
            except Exception as e:                           # e.g. a worker killed for lack of memory breaks the whole pool
                print(f"POOL BROKEN after {done} runs ({type(e).__name__}: {e}). Re-run the plan: finished runs are kept. "
                      "If it was memory, use fewer --processes.", flush=True)
                sys.exit(2)
            done += 1
            if err:
                failed.append(tag)
                print(f"[{done}/{len(todo)}] FAILED {tag}\n{err}", flush=True)
                continue
            acc = s["results"]["all"]["accuracy"]["rmse_iv"]
            viol = s["results"]["all"]["violations"]["any_violation_surface_rate"]
            print(f"[{done}/{len(todo)}] {tag}: RMSE {acc:.5f} violating {viol:.4f} ({secs:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"finished {done - len(failed)} runs, {len(failed)} failed, {(time.time() - t0) / 60:.1f} min", flush=True)
    for t in failed:
        print("  failed:", t)


if __name__ == "__main__":
    main()
