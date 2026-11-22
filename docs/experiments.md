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
test surfaces, over cells resolved in the truth and valid for every seed; MSE = bias^2 + variance exactly. This differs from the
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

## Changes after the fact

1. **H1 primary pair.** First set as absolute curvature error against absolute price error. A smoke test on one small run (before
   any main run) gave a price-error AUC of 0.26 for ridge, below chance, because violations sit in the wings where prices are
   tiny, so price-unit errors there are small however wrong the vol is. A price-unit comparison measures scale, so the primary
   pair became relative curvature error against pointwise implied vol error (the project's own accuracy unit). The original pair
   is still reported as the secondary comparison.
