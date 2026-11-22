# No-arbitrage violation detector (`src/violations.py`)

Step 3. Reproduce the calibration with `python scripts/calibrate_violation_tolerance.py` and the checks with
`python tests/test_violations.py`. Numbers come from `results/violation_tolerance.json`.

## What counts as a violation

Zero rates and dividends, spot 1. Each of the three Carr and Madan conditions is the price of a portfolio whose payoff can
never be negative, so a violation margin is that price when it is negative, **in price units**:

| Condition | Portfolio | Margin (positive means violated) |
| --- | --- | --- |
| Decreasing in strike | call spread | C(K[i+1]) - C(K[i]) |
| Convex in strike | butterfly (weights fit uneven strikes) | minus the butterfly price |
| Increasing in maturity | time spread | C(K, T[j]) - C(K, T[j+1]) |

A stencil is flagged when its margin exceeds the tolerance. The convexity margin is linked to probability: the slope
increment `s[i] - s[i-1]` is the implied probability mass around `K[i]`, and `negative_mass` adds up the negative ones
(probability units). `point_flags` turns stencil flags into one True or False per grid point, which is the cell-level
label to use when testing whether curvature error predicts violations (H1).

## Tolerance: how it was set and how it was checked

Rule: 10 times the largest margin seen on exact Heston surfaces in a calibration sample, rounded up to a power of ten.

- **Calibration sample:** all 32 corners of the parameter box plus 2,000 Sobol sets, on the dense grid (101 strikes, 20
  maturities), on the base 88-point grid, and on the base grid after converting prices to implied vols and back (the route a
  network's output takes). The largest margin was **2.3e-14**, so the tolerance is **1e-12** (about 44 times that).
- **Validation on fresh exact surfaces:** Sobol seed 202, uniform random sets, and a subset with at least one parameter within
  10 percent of its bound (like the planned boundary test subset), 31,727,679 checks in total. **False positives: 0.** A
  95 percent upper bound on the false-positive rate per check is about 1e-7 (rule of three). The largest margin on the fresh
  data was 4.5e-14, still 22 times below the tolerance.

One price-unit tolerance works on every grid because the noise in an exact price (about 1e-14 to 1e-13) does not depend on
the strike spacing.

## What the tolerance does not tell you: sensitivity depends on the grid and on where you look

The tolerance controls false alarms. How easily a model error becomes a violation is a different thing, and it depends on how
much room the true surface has before a violation, which is the true butterfly price. On exact surfaces (500 Sobol sets):

| Grid | Share of convexity stencils with true butterfly price below 1e-6 | At maturity 0.1 | At maturity 2.0 |
| --- | --- | --- | --- |
| Base (strike step 0.1) | 3.6 percent | 22.6 percent | 0.02 percent |
| Dense (strike step 0.01) | 8.7 percent (91 percent are below 1e-4) | 54.9 percent | 0.9 percent |

Injecting a single dent of known size into an exact dense-grid surface (300 sets, 20 random positions each) and asking
whether the convexity check flags it:

| Dent size | 1e-12 | 1e-9 | 1e-7 | 1e-5 | 1e-3 |
| --- | --- | --- | --- | --- | --- |
| Share flagged | 0.6 percent | 3.2 percent | 5.3 percent | 17.7 percent | 100 percent |

What follows for the experiments (these are descriptive facts about exact surfaces, not results about any network):

- The violation rate is not a property of the network alone. It depends on the grid: a finer strike step shrinks the true
  butterfly prices (they scale with the square of the step), so the same price error is easier to flag. Always compare
  models on the same fixed grid, and report the dense-grid and base-grid rates separately.
- Headroom is smallest at short maturities and in the wings. That is where a small error is most likely to flip the sign, which
  is the mechanism proposed in the deck (squared error scores levels, and the margin is tiny where the density is tiny).
  Whether it is where the networks actually fail is what H1 and H2 will test.

## Limits and assumptions

- **Evaluate in float64.** A float32 network has rounding noise of order 1e-7 on prices near 1, far above the tolerance, so
  float32 outputs would be flagged for rounding alone. (The reproduction already sets float64.)
- **Implied-vol route cannot see the deep wings.** Cells whose out-of-the-money price is below 1e-11 have no implied vol
  (NaN) and are not checked on that route: 1.9 percent of base-grid cells on the shipped parameter sets, mostly maturity 0.1,
  strikes 0.5 to 0.7. Checking prices on the dense grid avoids this, which needs a model that outputs prices.
- **Not checked:** the bounds C >= max(1 - K, 0), C <= 1 and the slope bound dC/dK >= -1. Exact surfaces satisfy them. They
  could be added if a network is seen to break them.
- **Grid ends.** Convexity is checked at interior strikes only; the first and last strike have no butterfly.
