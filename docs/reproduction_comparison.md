# Reproduction check: PyTorch port against the original notebook

Status: step 1 of the task list. Everything below is computed from files in this repository or from the authors'
public notebook. Items marked **unverified** could not be checked from this environment.

## What was compared

- Ours: `src/reproduce_heston.py` (PyTorch, seed 0), result in `results/reproduction.json`,
  per-cell analysis in `results/reproduction_error_analysis.json` and `results/reproduction_error_cells.png`
  (made by `scripts/analyse_reproduction.py`).
- Original: `Heston/NNHeston.ipynb` in amuguruza/NN-StochVol-Calibrations (MIT), downloaded from the master branch.
  It still contains its saved training log and its error figure (`HestonNNErrors.png` logic, cell 20).
- The paper itself (Horvath, Muguruza, Tomas, 2021, arXiv:1901.09647) **could not be fetched**: arxiv.org is not
  reachable from the work environment, and web search returned only the abstract. We therefore have **no check against
  numbers quoted in the paper text**. Someone with access should read the Heston section and add the figures here.

## Setup is the same

Same data file (12,000 rows), same 85/15 split (`random_state=42`, 10,200 train and 1,800 test), parameters scaled to
[-1, 1], surface standardised, 5-30-30-30-88 network with ELU, RMSE loss, Adam, batch 32, 200 epochs. The notebook uses
the test set as validation data, as our port does.

## Result: consistent with the original

Validation RMSE in standardised surface units, final epochs:

| | Epoch 200 | Mean of last 10 epochs | Range over last 10 epochs |
| --- | --- | --- | --- |
| Original notebook (Keras log) | 0.0298 | 0.0289 | 0.0276 to 0.0315 |
| Our PyTorch port | 0.0279 | not stored | not stored |

Our final value sits inside the original's epoch-to-epoch range, so the two runs are statistically indistinguishable on
this metric. We ran one seed only, so we cannot yet say how large the seed-to-seed spread is.

Relative implied-vol error on the 1,800 test surfaces (ours): mean 0.18 percent, median 0.12 percent, 99th percentile
1.34 percent, maximum 18.06 percent. The original notebook does not print a single overall number; it shows per-cell
heatmaps. Read by eye from its colour bars (approximate, **unverified to better than about 0.1 percentage points**):

| Quantity | Original (read from figure) | Ours |
| --- | --- | --- |
| Highest per-cell mean error | about 1.0 percent, at maturity 0.1, strike 0.5 | 0.99 percent, same cell |
| Per-cell mean error elsewhere | mostly 0.15 to 0.4 percent | 0.12 to 0.74 percent (most below 0.3) |
| Highest per-cell maximum error | about 16 percent, at maturity 0.1, strike 1.5 | 18.06 percent, same cell |

The error pattern matches: largest at the shortest maturity and at the strike edges, smallest at long maturities near
the money. The original's test split is identical, so the same surfaces are being scored, but its trained weights are
not in the repository we have and the Keras and PyTorch random streams differ, so exact agreement is not expected.

Conclusion: the reproduction is faithful. The gap to the original is within what a different training run produces;
no systematic difference found.

## The 18 percent worst case

- It is one cell, not a trend. Only 0.06 percent of the 158,400 test cells (about 93) exceed 5 percent, and 1.6 percent
  exceed 1 percent. The surface with the 18.06 percent error is at maturity 0.1, strike 1.5.
- Short maturity is where the errors concentrate: the three largest per-cell maxima in the table sit in the maturity-0.1
  row, and the mean error is highest there (0.99 percent at strike 0.5, against about 0.12 to 0.2 percent for
  maturities of 0.9 and above).
- Surfaces with parameters near the lower bounds are worse on average. Spearman correlation of per-surface mean error
  with the parameter: xi0 -0.35, rho -0.39, sigma -0.38, theta -0.22, kappa -0.11. The worst surface has xi0 = 0.0002
  (lower bound 0.0001), rho = -0.16, sigma = 0.055. So error rises towards the low-variance and low-vol-of-vol
  corner and for strongly negative rho.
- Relative error and true implied vol are only weakly related (Spearman -0.26), and the worst cell has a true vol of
  0.18, so this is not simply a division-by-tiny-vol artefact.

These are descriptive statistics on one trained network. They are a hypothesis for later, not a finding: if errors are
largest near the parameter bounds, that bears on H2 (interior against boundary). Do not cite it as evidence until it
has been repeated over seeds with confidence intervals.

## Caveats

1. One seed, one split.
2. Paper numbers not checked (see above).
3. Heatmap values for the original are read from an image.
4. Relative error is the original notebook's metric; the project's own metrics (absolute implied-vol error in vol
   points, curvature, violations) come later and are not affected by this check.
