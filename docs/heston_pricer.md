# Exact Heston pricer (`src/heston.py`): method and validation

Step 2 of the task list. Everything here is reproducible with `python scripts/validate_heston.py` and
`python tests/test_heston.py`. Numbers come from `results/heston_validation.json`.

## What it provides

- `sobol_params(n, seed)`: scrambled Sobol parameter sets over the base-code ranges (xi0, rho, sigma, theta, kappa).
- `call_prices`, `otm_prices`, `implied_vols`, `surface88`: exact prices and implied vols on the 88-point base grid
  (`STRIKES`, `MATURITIES`) or on any grid. `DENSE_STRIKES` (101 points, 0.5 to 1.5) and `DENSE_MATURITIES`
  (20 points, 0.1 to 2.0) are the default dense grid for convexity and calendar checks; both contain every base-grid
  strike and maturity. The dense grid is a choice we can change before step 3.
- Only numpy and scipy are needed (mpmath is optional, for the 30-digit cross-check in the validation script).

## Conventions, recovered from the shipped data

Spot 1, no interest rate or dividends (strike means K/S0), parameter order (xi0, rho, sigma, theta, kappa), with xi0
the initial variance, sigma the volatility of variance and **theta the long-run variance, not volatility**. Reading
theta as a volatility (squaring it) gives errors of up to 0.2 in implied vol; reading it as a variance reproduces the
shipped values to about 1e-12 at four of the eight maturities (below). Surfaces are flattened maturity-major.

## Method

Lewis/Gatheral Fourier integral with the "little Heston trap" characteristic function. Two numerical points matter:

1. `a - d` is computed as `-sigma^2 (z^2 + i z) / (a + d)` and the logarithm with `log1p`. The textbook form loses about
   five digits when sigma is small (it multiplies a cancellation error by kappa*theta/sigma^2, up to 2e4).
2. The integrand decays very slowly when xi0 is tiny and sigma is large (it needs u beyond 6,400 at some corners), so the
   integration range is chosen per parameter set and maturity, then parameter sets with the same range are batched.
   Composite 16-point Gauss-Legendre panels are used, narrow near u = 0 because 1/(u^2 + 1/4) has poles at u = +-i/2.
   (A first version with uniform panels had a constant error of about 4e-8 from exactly this; the validation caught it.)

Implied volatility is inverted from the out-of-the-money price (put below the forward, call above) by vectorised bisection.

## Validation results

| Check | Result |
| --- | --- |
| Against textbook formula with scipy quad, 56 parameter sets (32 corners and 24 Sobol), 3 maturities, 3 strikes | max absolute price difference 8.3e-12; median 8.9e-16 |
| Worst 6 of those cases at 30 digits (mpmath) | `heston.py` is within 2.8e-13; the double-precision reference is the one that is off, by 8.0e-12 |
| Exact surfaces at all 32 corners on the dense grid | finite; second difference in strike at least -3.6e-14, calendar difference at least -1.4e-14, strike difference at most 2e-15. This is the numerical noise floor of the exact surfaces and is the number to use when setting the violation tolerance in step 3 |
| Speed (one core) | about 10 minutes for 250k surfaces on the 88-point grid, about 24 minutes on the dense grid |

## Comparison with the shipped data (12,000 surfaces)

The shipped implied vols agree with exact prices to about 1e-12 at maturities 0.6, 1.2, 1.8 and 2.0, which confirms the
conventions and the pricer. At the other four maturities the shipped values are lower than exact, with a gap that
shrinks as maturity grows:

| Maturity | Mean (shipped minus exact) | RMS | 99th percentile of absolute gap |
| --- | --- | --- | --- |
| 0.1 | -4.6e-4 | 5.9e-4 | 1.2e-3 |
| 0.3 | -1.8e-4 | 2.2e-4 | 4.0e-4 |
| 0.9 | -4.1e-5 | 4.9e-5 | 1.1e-4 |
| 1.5 | -1.8e-5 | 2.2e-5 | 5.9e-5 |

(Implied vol units; 1e-3 is 0.1 volatility points. Excludes unresolved cells, below.)

What we know and do not know:

- The cause is **unknown**. Simple linear interpolation (in vol or in total variance) of exact values between the four
  exact maturities does not reproduce it (RMS 1e-3 to 2e-3, far larger than the actual gap), so that explanation is
  rejected. Do not state a cause in the write-up.
- Consequence for the reproduction: the PyTorch port was trained and scored against the shipped labels, so its 0.0009
  RMSE measures fit to those labels, not to the true Heston surface. At maturity 0.1 the label gap (RMS 5.9e-4) is of
  the same order as that overall RMSE. For the experiments we train and score on our own exact labels.

## Unresolved cells

An out-of-the-money price below `PRICE_FLOOR = 1e-11` is not resolved by this method in double precision (the exact
prices carry noise of about 2e-13, and 4,001 out-of-the-money prices on the shipped parameter sets come out slightly
negative). Such cells return NaN implied vol and are masked, not filled. On the shipped parameter sets that is 19,800 of
1,056,000 cells (1.9 percent), nearly all at maturity 0.1: strike 0.5 in 98.5 percent of surfaces, 0.6 in 42 percent,
0.7 in 13 percent and 1.5 in 5 percent (and 0.5 at maturity 0.3 in 6 percent).

The shipped file does contain a value in these cells, so they are exactly where its labels are least meaningful. The
original notebook's error heatmap peaks in this same corner (maturity 0.1, strike 0.5). That is suggestive, not shown:
we have not tested whether the network's error there is explained by label noise.

Open design decision for the experiments: how to treat NaN cells in the 88-vector (mask them in loss and metrics, or
drop surfaces with any NaN, which would remove 98 percent of the shipped set). Masking is the default recommendation.
