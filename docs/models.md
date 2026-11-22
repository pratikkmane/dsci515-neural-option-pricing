# Baselines and neural surrogates (step 4)

Code: `src/data.py`, `src/evaluate.py`, `src/baselines.py`, `src/networks.py`. Runners: `scripts/run_baselines.py`,
`scripts/train_networks.py`. Tests: `tests/test_baselines.py` (no torch needed), `tests/test_networks.py` (needs torch).
Result files: `results/baselines/*.json` and `results/models/*.json` (weights `*.pt` are git-ignored).

**Every number in this note is preliminary: one seed, one training set per setting, no confidence intervals.** They show that the
pipeline works and where to look. They are not findings, and nothing here tests H1, H2 or H3.

## Setup

- **Data** (`src/data.py`): exact Heston surfaces on the 88-point base grid. A training set is the first n rows of a scrambled
  Sobol sequence, so smaller sets are prefixes of larger ones for the same seed; the seed picks the scramble (a different
  training set). The test set is fixed: 20,000 Sobol sets with their own seed. Region label: **interior** means every
  parameter is in the central 80 percent of its range (33 percent of the test set), **boundary** means at least one is in
  the outer 10 percent at either end (67 percent). Unresolved cells (price below 1e-11, about 4 percent of cells over the whole
  box) are NaN: masked, never filled.
- **Label noise** is multiplicative Gaussian, `y (1 + level z)`, with levels 0, 0.1, 0.5, 2 percent, drawn independently per
  cell and applied to the training labels only. Test labels are always exact.
- **Evaluation** (`src/evaluate.py`): implied vol RMSE, 99th percentile error and relative error on resolved cells; violation
  rates from `src/violations.py` (tolerance 1e-12) on the base grid, overall and by region. Predictions that cannot be turned
  into prices (NaN or non-positive vol, or an uninvertible price) are counted as `unusable_cell_share` and shown next to the
  rates, so a model cannot look consistent by producing nonsense that is skipped.

## Models

| Model | What it is |
| --- | --- |
| `ridge_poly` | Polynomial features of the 5 scaled parameters, degree (2 to 8) and penalty chosen by 3-fold cross-validation |
| `gradient_boosting` | One gradient-boosted model per principal-component score |
| `gaussian_process` | One GP (ARD RBF kernel) per score, **capped at 3,000 training points** (exact GPs cost O(n^3)); the cap is stored with every result |
| Grid network | The base paper's: 5-30-30-30-88, ELU, RMSE loss on the standardised surface, Adam, batch 32. Masked loss on unresolved cells |
| Pointwise network | Input (5 parameters, strike, maturity), one output: **price** (as in the proposal) or **implied vol** (price by Black-Scholes). Strike slope and curvature (the implied density) by autograd |

PCA models: linear PCA needs about 19 components for a truncation error of 5e-4 in implied vol on this data (more on noisy
labels: it reached the cap of 40 with 5 percent noise), far more than the 10 first tried, which left a truncation error of
1.7e-3 or more, above the network's own error. The count is chosen
automatically from that target and the achieved truncation error is stored. The GP kernel needed lower bounds on the length
scales: without them two components collapsed to a noise-fitting kernel.

Missing cells are filled by a low-rank reconstruction **for fitting the baselines only**; networks use a masked loss instead.

Checks that the autodiff is right (`tests/test_networks.py`): for a constant-vol network the autograd price, slope and density
equal the Black-Scholes closed forms to 1e-9 to 1e-12 (this checks every scaling of strike and maturity); autograd agrees with
finite differences on a random network; a ReLU price network has exactly zero curvature.

## Preliminary results (seed 0)

Violating surfaces = share of the 20,000 test surfaces with at least one flagged stencil, base grid. `conv` and `cal` are the
convexity and calendar-spread conditions (the monotone condition is essentially never violated by grid models).

| Model | n | Label noise | RMSE (vol) | p99 abs err | Violating: all | interior | boundary | conv | cal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_poly | 2,000 | 0 | 0.00307 | 0.01274 | 2.8% | 0.6% | 3.9% | 0.4% | 2.7% |
| ridge_poly | 2,000 | 0.5% | 0.00311 | 0.01278 | 3.5% | 0.6% | 4.9% | 1.0% | 2.7% |
| ridge_poly | 2,000 | 2% | 0.00372 | 0.01369 | **31.2%** | 6.1% | 43.5% | 28.6% | 4.9% |
| ridge_poly | 10,000 | 0 | 0.00211 | 0.00832 | 2.2% | 0.6% | 3.0% | 0.1% | 2.2% |
| gradient_boosting | 2,000 | 0 | 0.00515 | 0.01772 | 1.9% | 0.3% | 2.7% | 0.2% | 1.9% |
| gradient_boosting | 2,000 | 0.5% | 0.00516 | 0.01780 | 2.0% | 0.3% | 2.8% | 0.2% | 2.0% |
| gradient_boosting | 2,000 | 2% | 0.00539 | 0.01861 | 8.2% | 2.5% | 10.9% | 6.3% | 2.2% |
| gradient_boosting | 10,000 | 0 | 0.00295 | 0.00940 | 1.5% | 0.2% | 2.1% | 0.0% | 1.5% |
| gaussian_process | 2,000 | 0 | 0.00235 | 0.00837 | 1.8% | 0.4% | 2.5% | 0.1% | 1.8% |
| gaussian_process | 10,000 (3,000 used) | 0 | 0.00229 | 0.00830 | 1.8% | 0.3% | 2.5% | 0.0% | 1.8% |
| Grid network | 2,000 | 0 | 0.00163 | 0.00553 | 0.3% | 0.0% | 0.4% | 0.0% | 0.3% |
| Grid network | 2,000 | 0.5% | 0.00166 | 0.00571 | 0.4% | 0.0% | 0.5% | 0.0% | 0.4% |
| Grid network | 2,000 | 2% | 0.00210 | 0.00651 | 2.0% | 0.0% | 3.0% | 1.6% | 0.6% |
| Grid network | 10,000 | 0 | 0.00118 | 0.00348 | 0.1% | 0.0% | 0.1% | 0.0% | 0.1% |
| Pointwise, vol output, softplus | 2,000 | 0 | 0.00729 | 0.02543 | 1.3% | 0.0% | 1.9% | 0.4% | 0.9% |
| Pointwise, price output, softplus | 2,000 | 0 | 0.02872 | 0.09594 | 92.4% | 95.3% | 91.0% | 84.7% | 78.8% |
| Pointwise, price output, ReLU | 2,000 | 0 | 0.08515 | 0.38179 | 97.7% | 98.2% | 97.4% | 86.6% | 77.9% |

Pointwise rows: width 64, depth 3, 40 epochs, batch 1024. Pointwise price-output rows also have `unusable_cell_share` of 23.8
percent (softplus) and 0.4 percent (ReLU), meaning predicted out-of-the-money prices too inaccurate to give an implied vol.

## What these runs suggest (to be tested properly in step 5, not claims)

- **Accuracy and consistency can move apart.** From no noise to 2 percent label noise, RMSE rose 21 to 29 percent for ridge and
  the grid network (5 percent for gradient boosting) while the share of violating surfaces rose from 2.8 to 31 percent (ridge),
  0.3 to 2.0 percent (grid network) and 1.9 to 8.2 percent (boosting). The rise is in the convexity condition. At 0.5 percent noise
  they barely move (ridge 2.8 to 3.5 percent, boosting 1.9 to 2.0, grid network 0.3 to 0.4). This is the pattern H2 describes, in one seed.
- **Where.** In these runs the grid network has no violating interior surface at any setting (0 of 6,559 interior surfaces in each of the four
  runs), with violations in the boundary region only; ridge and boosting show interior violations under noise (6.1 and 2.5 percent). H2 says violations occur
  inside the training range, so the grid network at this size does not (yet) show it; the baselines do.
- **Size.** Ridge, boosting and the grid network improve in RMSE from 2,000 to 10,000 surfaces and their violation rates fall or
  stay level. The GP cannot benefit because it is capped at 3,000 points.
- **Calendar violations at zero noise** are 1.5 to 2.7 percent for the baselines and mostly in the boundary region. Why is
  not investigated.
- **Violations are small in size.** Mean margin when violated is about 1e-6 to 2e-4 (price units, spot 1) for the grid network
  and about 1e-3 for ridge at 2 percent noise (largest 1.6e-2). All are far above the 1e-12 tolerance, so they are not numerical
  noise, but they are small in money terms. The write-up should say so and may want a severity cut (for example counts above
  1e-6 and 1e-4) next to the tolerance-based rate.
- **Pointwise price output is much harder** at this small budget: most surfaces break the monotone or calendar rule and a
  quarter of resolved cells cannot be inverted for the softplus network. A squared-error loss on price levels ignores the wings
  where prices are tiny, which is the mechanism in the deck. The vol-output pointwise network is far better. These networks are
  badly under-trained here (40 epochs), so this says nothing about what they can reach.

## Compute (measured on this container, 4 cores, runs overlapping)

Ridge under 1 s (3 s at 10,000), boosting 6 s and 13 s, GP 341 s and 814 s (3,000-point cap), grid network 9 s (2,000, 200
epochs) and 91 s (10,000), pointwise 29 to 48 s (2,000, 40 epochs). Extrapolating the grid network linearly in n at batch 32,
250,000 surfaces would take roughly 40 minutes per run (an estimate, not measured), so the largest sizes should use a larger
batch and the number of seeds there will have to be reduced.

## Decisions for the team

1. **Pointwise output.** The proposal says price (so ReLU has zero curvature); a vol output behaves very differently. Suggest
   running both and presenting the representation comparison as price against vol as well as grid against pointwise.
2. **Noise on what.** Noise is applied to the training labels of whatever the model fits (vols for grid and vol-output nets,
   prices for price-output nets). Say so in the write-up, since 2 percent on a price and on a vol are not the same disturbance.
3. **Severity.** Report violation size as well as presence (see above).
4. **GP size.** The cap of 3,000 is a compute decision; quote it with every GP result.
5. **Training budget for pointwise nets** must be set before comparing them with grid networks.

## Not in this step

The derivative loss and the arbitrage penalty (step 6); exact strike-derivative labels for supervising slopes and for the
curvature-error metric (they can be obtained from the same Fourier integral and are not built yet); bootstrap intervals, the
bias-variance split and the AUC analysis (step 5).
