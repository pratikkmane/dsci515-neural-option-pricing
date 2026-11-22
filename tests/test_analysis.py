"""Checks for the statistics in scripts/analyse_experiments.py. Run with `python tests/test_analysis.py`."""
import os, sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
sys.path.insert(0, os.path.join(here, "..", "scripts"))
import analyse_experiments as A


def test_bias_variance_adds_up_exactly():
    rng = np.random.default_rng(0)
    truth = rng.random((50, 88)) + 0.1
    preds = truth[None] + 0.01 * rng.standard_normal((6, 50, 88)) + 0.003            # a bias of 0.003, noise 0.01 across seeds
    preds[:, 3, 5] = np.nan                                                             # a cell invalid for every seed: dropped
    truth2 = truth.copy(); truth2[7, 7] = np.nan                                        # unresolved in the truth: dropped
    r = A.bias_variance(preds, truth2)
    assert np.isclose(r["mse"], r["bias2"] + r["variance"], rtol=1e-12) and r["cells"] == 50 * 88 - 2
    # with 6 seeds: plug-in bias^2 = 0.003^2 + 0.01^2 / 6 and plug-in variance = 0.01^2 * 5 / 6, up to sampling noise
    assert abs(r["bias2"] / (0.003 ** 2 + 0.01 ** 2 / 6) - 1) < 0.1 and abs(r["variance"] / (0.01 ** 2 * 5 / 6) - 1) < 0.05
    assert abs(r["bias2_corrected"] - 0.003 ** 2) < 3e-6 and abs(r["variance_unbiased"] / 0.01 ** 2 - 1) < 0.05
    assert A.bias_variance(preds[:1], truth)["cells"] == 0                              # one seed has no variance


def test_boot_rates_are_centred_and_ordered():
    rng = np.random.default_rng(1)
    f = rng.random((10, 4000)) < 0.1
    mask = np.ones(4000, bool)
    r = A.boot_rates([f, f], [mask, mask], np.random.default_rng(2), n_boot=500)
    m, lo, hi = A.interval(r[0])
    assert abs(m - f.mean()) < 0.01 and lo < f.mean() < hi
    assert np.array_equal(r[0], r[1])                                                  # common resampling for paired settings
    allone = A.boot_rates([np.ones((5, 100), bool)], [np.ones(100, bool)], np.random.default_rng(3), n_boot=50)
    assert np.allclose(allone, 1.0)


def test_boot_rates_respect_the_mask():
    f = np.zeros((6, 1000), bool); f[:, :100] = True                                    # violations only in the first 100 surfaces
    inside = np.zeros(1000, bool); inside[:100] = True
    r = A.boot_rates([f, f], [inside, ~inside], np.random.default_rng(4), n_boot=200)
    assert np.allclose(r[0], 1.0) and np.allclose(r[1], 0.0)


def test_seed_boot_edge_cases():
    assert np.isnan(A.seed_boot([np.nan, np.nan], np.random.default_rng(0))[0])
    assert A.seed_boot([0.3], np.random.default_rng(0)) == (0.3, 0.3, 0.3)
    m, lo, hi = A.seed_boot([1.0, 2.0, np.nan, 3.0], np.random.default_rng(0))
    assert m == 2.0 and lo <= m <= hi


def test_paired_difference_has_smaller_interval_than_unpaired_logic_would_give():
    rng = np.random.default_rng(5)
    base = rng.random((10, 3000)) < 0.2
    shifted = base | (rng.random((10, 3000)) < 0.05)                                    # a, b strongly correlated
    ra, rb = A.boot_rates([base, shifted], [np.ones(3000, bool)] * 2, np.random.default_rng(6), n_boot=600)
    m, lo, hi = A.interval(rb - ra)
    assert lo > 0                                                                       # the small real increase is detected when paired


def test_spearman():
    assert np.isclose(A.spearman(np.array([1, 2, 3, 4.0]), np.array([10, 20, 35, 80.0])), 1.0)
    assert np.isnan(A.spearman(np.array([1, 1, 1.0]), np.array([1, 2, 3.0])))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
