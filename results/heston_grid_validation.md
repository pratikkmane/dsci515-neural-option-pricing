# Heston Reference Pricing and Grid-Mapping Validation

## Objective

Validate an independent Heston European call pricer and determine how the original dataset stores its 88 implied-volatility outputs across 11 strikes and 8 maturities.

## Implementation

Implemented `src/heston.py` using:

- Heston characteristic-function pricing with numerical integration.
- Black–Scholes European call pricing.
- Black–Scholes implied-volatility inversion using Brent's method.

Implemented `src/validate_grid_mapping.py` to compare the independent reference prices and implied volatilities against the original dataset.

## Validation Results

### 1. Black–Scholes Limiting Case

- Heston call price: 0.07965170
- Black–Scholes call price: 0.07965567
- Absolute difference: 0.00000397

The results agree closely in the tested low-volatility-of-variance limit.

### 2. Characteristic-Function Consistency

- φ(0) = 1 + 0j
- φ(−i) = 1 + 0j, with zero interest rate

Both identities passed for the tested parameters.

### 3. Strike–Maturity Mapping

Compared two possible arrangements of the 88 implied-volatility outputs.

| Metric | Maturity-first | Strike-first |
|---|---:|---:|
| Valid comparison points | 87/88 | 87/88 |
| Mean absolute IV error | 0.00902590 | 0.03656469 |
| Maximum absolute IV error | 0.03194989 | 0.14108788 |

**Finding:** Maturity-first ordering is strongly supported, meaning strikes likely vary fastest within each maturity.

Under this provisional mapping, output index 10 corresponds to strike 1.5 and maturity 0.1 years.

### 4. Numerical Limitations

One reference implied-volatility calculation failed at maturity 0.1 and strike 0.5 because the calculated call price fell approximately 1 × 10⁻⁹ below its theoretical lower bound.

The largest reference IV discrepancies occurred at short maturities and extreme strikes.

Increasing the integration upper bound from 200 to 500 and tightening integration tolerances did not materially improve the results.

### 5. Reference Price Comparison

| Maturity | Strike | Reference minus dataset-implied call price |
|---|---:|---:|
| 0.1 | 0.6 | +0.0000000334 |
| 0.1 | 1.0 | +0.0001247224 |
| 0.1 | 1.5 | −0.0000012603 |
| 0.3 | 0.5 | +0.0000190711 |

Small call-price differences can produce relatively large implied-volatility differences when Black–Scholes vega is very low.

## Conclusions

The independent Heston pricer passed initial mathematical consistency tests. Numerical evidence strongly favours maturity-first output ordering, but the original dataset-generation conventions have not been fully verified.

These tests do not establish that the neural network violates no-arbitrage conditions.

## Next Steps

1. Investigate the original Heston dataset-generation method and pricing conventions.
2. Confirm the strike–maturity ordering independently.
3. Implement strike monotonicity and convexity tests on option prices.
4. Compare no-arbitrage violations between the dataset and neural-network predictions.
5. Document numerical tolerances and distinguish genuine violations from numerical artefacts.