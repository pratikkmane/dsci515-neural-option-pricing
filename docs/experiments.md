# Experiments (step 5): protocol, decisions and interpretation

This note was written **before the main runs finished**, so the decisions and the analyses are fixed in advance. Changes made
afterwards are listed under "Changes after the fact". Results are generated into `docs/experiments_results.md` by
`scripts/analyse_experiments.py`; this file is the protocol and the reading guide. The derivative loss and arbitrage penalty
(H3) are step 6 and are not here.

## Decisions on the open points from step 4

1. **Pointwise output: run both.** Price output (the proposal's version, where ReLU gives zero curvature) and implied vol output
   (price by Black-Scholes), each with softplus and ELU, and ReLU for price output.
2. **Noise on what.** Multiplicative Gaussian noise on the training labels of whatever the model fits: implied vols for the grid
   network and the baselines, call prices for price-output nets, implied vols for vol-output nets. A price disturbance and a
   vol disturbance of the same percentage are not the same size of error, so noise levels are comparable within a model family,
   not across the grid/price/vol families. Test labels are always exact.
3. **Severity.** Every result carries the share of surfaces whose largest violation exceeds 1e-8, 1e-6 and 1e-4 (price units,
   spot 1) next to the 1e-12 tolerance rate.
4. **Gaussian process size.** Capped at 3,000 training points, stated with every GP result. Only two sizes and three seeds are
   run, because the cap makes larger sizes identical to 3,000 and the fits take minutes.
5. **Pointwise training budget** was chosen on a separate validation set (not the test set) before any pointwise test result was
   looked at: see "Pointwise budget".

## Pointwise budget (validation pilot)

Pilot on a separate validation set (2,000 surfaces, its own seed), training on 5,000 surfaces, softplus, one seed, four budgets
per output type (`scripts/pilot_pointwise_budget.py`, `results/pilot_pointwise.json`). Rule fixed before running: per output
type, the configuration with the lowest validation error of that output, preferring a cheaper one within 5 percent.

| Output | Configuration | Validation RMSE (vol) | Validation price RMSE | Unusable cells | Violating surfaces | Fit time |
| --- | --- | --- | --- | --- | --- | --- |
| vol | 64 x 3, 30 epochs, constant lr | 0.00402 | 0.00084 | 0.2% | 1.5% | 82 s |
| vol | 64 x 3, 30 epochs, cosine | 0.00711 | 0.00147 | 0.4% | 1.5% | 83 s |
| vol | **128 x 4, 30 epochs, cosine** | **0.00247** | **0.00040** | 0.1% | 0.1% | 249 s |
| vol | 64 x 3, 60 epochs, cosine | 0.00333 | 0.00067 | 0.2% | 0.9% | 158 s |
| price | 64 x 3, 30 epochs, constant lr | 0.02871 | 0.00354 | 24.4% | 93.4% | 81 s |
| price | 64 x 3, 30 epochs, cosine | 0.05688 | 0.00345 | 13.8% | 90.1% | 79 s |
| price | 128 x 4, 30 epochs, cosine | 0.04462 | 0.00200 | 10.6% | 99.6% | 169 s |
| price | **64 x 3, 60 epochs, cosine** | 0.04508 | **0.00182** | 13.3% | 98.7% | 115 s |

Chosen: vol output 128 x 4, 30 epochs, cosine; price output 64 x 3, 60 epochs, cosine (used for every activation). Caveats: one
seed; learning-rate schedule and network size were varied together, so this does not show that a cosine schedule helps (at equal
size it did worse for vol output). **Price-output networks stayed inconsistent at every budget tried**, with most surfaces
violating and 10 to 24 percent of cells not invertible to an implied vol, so that result is unlikely to be an artefact of this
particular budget, though a much larger budget was not tried.

## Design

Every run is one model, one training set (the first n rows of a scrambled Sobol sequence selected by the seed, so seeds are
independent training sets and sizes are nested), one noise level, one initialisation. The test set is fixed (20,000 surfaces,
interior and boundary parts as defined in `docs/models.md`).

| Part | Models | Sizes | Noise | Seeds |
| --- | --- | --- | --- | --- |
| core_small | base-paper grid network, ridge | 2k, 10k, 50k | 0, 0.1, 0.5, 2 percent | 10 |
| core_large | same | 250k | same | as many as time allows |
| boosting | gradient boosting | 2k, 10k, 50k | 0, 2 percent | 10 |
| gp | Gaussian process (3,000-point cap) | 2k, 10k | 0, 2 percent | 3 |
| architecture | grid network: depth 2, 4, 6; width 64, 128, 256; softplus; ReLU (one at a time around 3 x 30 ELU) | 10k | 0, 2 percent | 5 |
| pointwise | pointwise nets: vol output (softplus, ELU), price output (softplus, ELU, ReLU) | 10k | 0, 2 percent | 5 |

Grid-network budget: 200 epochs at the base paper's batch size of 32 up to 10,000 surfaces, and a batch size growing with n above
that (160 at 50,000, 800 at 250,000), so that every size gets about the same number of optimiser steps. Size is therefore
confounded with batch size above 10,000; the learning rate is constant, as in the base paper.

## Pre-specified analyses

All intervals are 95 percent percentile bootstrap intervals. Seeds are independent training sets, so resampling seeds captures
training-set variation; rates over test surfaces also resample the surfaces (two-stage), and differences between settings are
paired by seed and surface. There is no correction for multiple comparisons.

**H2 (variance, not only extrapolation).** For each model: (a) the interior violation rate with its interval, compared with the
detector's false-positive bound (2.2e-5 per surface); (b) paired differences in the interior rate between the smallest and largest
size at each noise level, and between noise 0 and 2 percent at each size; (c) the Spearman correlation across the (size, noise)
settings between the variance term and the interior violation rate, with a seed-bootstrap interval. H2 is read as supported where
the interior rate is clearly above the bound, rises as data shrink and noise grows (differences with intervals excluding zero), and
the variance term tracks it; it is read as refuted where violations appear only in the boundary region.

**Bias and variance.** For each setting, across seeds, squared bias and variance of the predicted implied vol on the first 2,000
test surfaces, over cells resolved in the truth and valid for every seed; MSE = bias^2 + variance exactly for the plug-in values. With S seeds the plug-in bias^2 is too large by (unbiased variance)/S and the
plug-in variance too small by a factor (S - 1)/S, so corrected values are reported alongside (S = 10 means a 10 percent effect on
the split, none on the total). This differs from the
proposal's "bootstrap-trained models": because exact data can be generated without limit, independent training sets (one per
seed) give the variance term directly and are cleaner than resampling one training set.

**H1 (accuracy hides shape).** Unit of analysis: a convexity stencil of the base grid (an interior strike at one maturity), positive
if the model violates convexity there. Four local error measures are always reported, each from the stencil's three points:
pointwise price error (absolute), pointwise implied vol error (absolute), butterfly (curvature) error (absolute), and relative
curvature error (the butterfly error over the exact butterfly price). **Primary comparison: relative curvature error against
pointwise implied vol error**, as AUC for ranking stencils, with the paired seed-bootstrap interval of the difference; H1 is read as
supported where the interval of the difference is above zero. Runs with fewer than 30 positive stencils give no AUC.

## Reading guide and cautions

- **H1 is close to true by construction.** A stencil violates convexity only if the model's butterfly price is negative, which
  needs the curvature error to be negative and larger than the exact butterfly price. So a large relative curvature error is
  necessary for a violation, and its AUC will be high whatever the model. The informative result is how far above the pointwise
  error it is, and whether pointwise error alone already finds the violations; do not present a high curvature AUC as a discovery.
- **Interior and boundary** are defined by the parameter box (interior: every parameter in the central 80 percent of its range,
  33 percent of the test set), not by where the training data are dense. Sobol training data cover the whole box, so "boundary"
  is a region of sparser neighbouring data, not extrapolation.
- **Violation rates depend on the grid** (docs/violations.md). All main results use the base 88-point grid; the dense grid is used
  for pointwise networks only.
- **Seeds vary the training set and the initialisation together.** The test set is fixed; its sampling variation enters only
  through the surface bootstrap.
- **Training budget is not tuned per model** (except the pointwise pilot). A model that looks inconsistent may be under-trained,
  and a larger budget might change it. The architecture runs address only part of that.
- **Violations are small in money terms** (see severity); a high rate does not mean large arbitrage profits.

## Changes after the fact (see also the deviations at the end)

1. **H1 primary pair.** First set as absolute curvature error against absolute price error. A smoke test on one small run (before
   any main run) gave a price-error AUC of 0.26 for ridge, below chance, because violations sit in the wings where prices are
   tiny, so price-unit errors there are small however wrong the vol is. A price-unit comparison measures scale, so the primary
   pair became relative curvature error against pointwise implied vol error (the project's own accuracy unit). The original pair
   is still reported as the secondary comparison.

## Deviations from the plan

2. **Fewer seeds where compute was short.** Planned 10 seeds everywhere. Run: 10 for sizes 2k to 50k (grid network, ridge, boosting);
   5 for 250k, for the architecture variants and for the pointwise networks; 3 for the Gaussian process. Paired comparisons use the
   seeds two settings share. Intervals for the 5-seed and 3-seed parts are correspondingly wide and rest on few draws.
3. **Subsets of the factor grid.** Boosting at noise 0 and 2 percent only; Gaussian process at 2k and 10k only (capped at 3,000
   points); architecture and pointwise at 10k only, noise 0 and 2 percent. The loss factor is step 6.
4. **Pointwise pilot.** A first, larger pilot (10,000 surfaces, 40 to 100 epochs, width 128, depth 4) was stopped after about an hour
   because it was too slow, and replaced by the smaller one reported above. The selection rule was written before either result.
5. **Ridge cost caps** introduced after the first runs: cross-validation on at most 40,000 rows, degrees limited to at most 800
   features (so degree 7 at most), and the final fit accumulated in blocks. The blocked fit reproduces the earlier fit to 1e-14.
6. **Crashes and re-runs.** Out-of-memory kills broke the process pool twice: the first pointwise attempt completed no runs (the
   autograd curvature check used chunks that were too large) and three Gaussian-process runs were lost. Both were re-run in
   full with smaller chunks or fewer processes; no completed result was discarded or selectively kept.
7. **Bias and variance** are from independent training sets, not bootstrap resamples (stated above).

## Findings

482 runs: 240 (grid network and ridge, 2k to 50k), 40 (250k), 60 (boosting), 12 (GP), 80 (architecture), 50 (pointwise). All numbers
below are from `docs/experiments_results.md` and `results/tables/` (re-create with `python scripts/run_experiments.py <plan>` for the
plans in `src/experiment.py`, then `python scripts/analyse_experiments.py`). Intervals are 95 percent bootstrap intervals over seeds
and test surfaces. These are descriptive results with no correction for multiple comparisons; read effects that are large and
consistent, not single intervals. The hypotheses are discussed at the end of each point.

### 1. Accuracy and consistency are only loosely tied

Across all settings, higher RMSE goes with more violating surfaces, but at a given accuracy the violation rate varies by more than
an order of magnitude: among the 23 settings (not pointwise networks) with RMSE between 1.5e-3 and 2.2e-3, the share of violating surfaces
ranges from 0.2 to 9.4 percent (`results/figures/accuracy_vs_violation.png`). A pricer's average error does not say whether it keeps the rules.

### 2. What the model is trained to output matters most

At 10,000 surfaces, pointwise networks that output the **call price** (squared error on prices) reach a price RMSE of only 0.3e-3 to
1.3e-3, yet violate the rules on **92 to 99 percent of surfaces**, and 6 to 12 percent of their cells cannot even be inverted to an
implied vol; their implied vol RMSE is 22e-3 to 44e-3, against about 1e-3 for the grid network. Networks that output the **implied
vol** with the same inputs violate on 0.04 to 0.13 percent of surfaces at the base grid (0.24 to 0.28 percent on the dense grid)
with vol RMSE 0.5e-3 to 1.4e-3. This is the mechanism in the deck made concrete: squared error on prices scores the levels that
dominate (near the money) and nearly ignores the wings where prices are tiny and the shape is decided. Caveats: five seeds, one
size; the price and vol networks have different budgets (each the best of four in the pilot); the pilot shows price output
staying inconsistent at every budget tried but not at a much larger one; and the vol-output networks (128 x 4) are larger than the
grid network (30 x 3), so "pointwise" against "grid" is not a capacity-matched comparison.

### 3. Label noise hurts consistency much more than accuracy, and data removes the effect

Grid network (base architecture), noise 0 to 2 percent: at 2,000 surfaces RMSE rises 30 percent (1.65e-3 to 2.14e-3) while the
violating share rises from 0.3 to **4.6 percent** [2.9, 6.8]; at 10,000 from 0.1 to 1.5 percent; at 50,000 it is 0.2 percent
and at 250,000 0.10 to 0.14 percent, essentially no change. Ridge: 3.0 to 23 percent at 2,000, 1.9 to 3.9 percent at 50,000, 1.9 to 1.9 percent
at 250,000. The extra violations are convexity violations (ridge at 2,000: 0.4 to 20.5 percent), and they are not tiny:
18.9 percent of ridge surfaces exceed 1e-4 in violation size at that setting, 3.1 percent for the grid network. The Gaussian
process fitted to noisy labels is the extreme: 27 to 32 percent of surfaces violate (interior 30 to 35 percent) with RMSE only
31 to 37 percent above its noise-free value. Caveat: the noise levels are per cell and multiplicative on the quantity each model
fits (docs/models.md), so they are comparable within a model family, not across families.

### 4. More data does not fix a misspecified model

At zero noise the violating share barely falls with data for the baselines: ridge 3.0 percent (2k) to 1.9 percent (250k); boosting
2.1 to 1.1 percent (50k); GP 1.9 percent at 2k and 10k. For ridge, the corrected squared bias is 3.8e-6 at 250,000 surfaces while
the variance across seeds has fallen to 7.7e-9: its remaining violations (mostly the calendar condition, interior 0.4 percent) are
bias, not estimation variance. The grid network is different: at zero noise it violates on 0.3 percent (2k) to 0.1 percent (250k)
of surfaces, its bias and variance terms are of similar size (1.2e-6 and 1.5e-6 at 2k) and both shrink with data.

### 5. H2 (variance, not only extrapolation): supported in a narrow regime

"Interior" is every parameter in its central 80 percent range (6,559 test surfaces). For the **base grid network**: with clean or
lightly noisy labels (0 to 0.5 percent) there is **no violating interior surface** in any run (upper bound 0.005 percent per
surface at 10 seeds), with violations confined to the boundary region (0.1 to 0.6 percent). Interior violations appear only at 2
percent noise and small data: 0.91 percent at 2,000 surfaces and 0.16 percent at 10,000, and none at 50,000 or 250,000. The paired
differences are positive (noise 2 percent minus 0 at n=2,000: +0.95 points [+0.23, +1.95]; 2,000 minus 50,000 at 2 percent noise:
+0.93 points [+0.22, +2.01]), so there the violations rise as data shrink and noise grows, as H2 says. For ridge, boosting and
the GP there are interior violations at every setting (ridge 0.4 to 0.7 percent at zero noise; GP and boosting similar, GP 30 to 35
percent with noise). **Verdict:** for the base network H2 holds only when labels are noisy and data are scarce; with clean labels
the criterion "refuted if violations appear only near the bounds" is met. **Variance and violations move together** (Spearman
correlation between the variance term and the interior violation rate across the size and noise settings: boosting +1.00
[+0.94, +1.00], ridge +0.66 [+0.42, +0.84], grid network +0.58 [+0.42, +0.58], 5 to 10 seeds). This is weak evidence for a variance
mechanism: both quantities fall with data and rise with noise, so the correlation is partly a shared dependence on those two factors.

### 6. Capacity interacts with noise

At 10,000 surfaces and 2 percent noise, wider networks violate far more than the base (30 x 3): 6.0 percent [2.6, 11.6] for width
128 and 9.1 percent [3.7, 17.6] for width 256, against 1.5 percent [0.7, 2.3], with interior rates 2.7 and 4.3 percent against
0.2 percent, while their RMSE is no better (1.61e-3 and 1.71e-3 against 1.52e-3). With clean labels every variant is consistent
(at most 0.2 percent of surfaces). Depth (2, 4, 6) and the activation (softplus, ReLU) show no clear effect at 5 seeds. This is the
pattern expected if larger networks fit label noise, but it rests on 5 seeds and one size.

### 7. H1 (accuracy hides shape): true by construction, and informative only under noise

Primary comparison: AUC of relative curvature error against AUC of pointwise implied vol error for ranking the convexity stencils
that violate. Relative curvature error has AUC at least 0.97 in every cell with enough violations, **by construction** (a violation
needs a large relative curvature error). The informative number is the pointwise error. **With clean labels pointwise vol error
already ranks violations almost as well** (AUC 0.996 to 0.998; primary difference only 0.002 to 0.003). **With 2 percent noise it
does not** (AUC 0.70 for boosting at 2,000, 0.81 GP, 0.84 grid network, 0.86 ridge), and the difference at 2,000 surfaces is 0.14 to 0.30 (0.03 to 0.17 at larger sizes) with
intervals above zero: under noise, violations come from small wiggles that a pointwise error does not flag. So the formal H1
criterion (interval of the difference above zero) is met nearly everywhere, but the practical gap is large only under noise. One
exception: for softplus price-output pointwise networks the difference is slightly negative (-0.007 to -0.008, intervals below
zero), where violations are almost everywhere. The 250,000-surface runs have too few violations per run for an AUC (under 30).

## Limits

Five seeds (three for the GP) leave wide intervals in the 250k, architecture, pointwise and GP parts. Nothing is corrected for the
many comparisons. One fixed test set of 20,000 surfaces. Violation rates depend on the grid (docs/violations.md): the base
88-point grid is used for all models, the dense grid only for the pointwise networks. Violations are mostly small in money terms
(except in the noisy small-data settings above). Training budgets are not tuned per model, so a model that looks inconsistent may
be under-trained. The noise definitions differ across model families. Interior and boundary are defined by the parameter box, not by
training-data density. H3 (derivative loss against arbitrage penalty) is not tested here: it is step 6.
