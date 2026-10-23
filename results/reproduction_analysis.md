# Heston Neural Network Reproduction: Error Analysis

## Objective

Investigate whether the neural implied-volatility surrogate exhibits local prediction weaknesses despite achieving low average prediction error.

## Dataset

- 12,000 Heston parameter combinations
- 5 input parameters
- 88 implied-volatility outputs per surface
- 10,200 training samples and 1,800 held-out test samples

## Reproduction Performance

- Mean relative error: 0.1818%
- Maximum relative error: 18.0627%
- Implied-volatility RMSE: 0.0009302

## Initial-Variance Quartile Analysis

| Quartile | xi0 Range | Mean Maximum Relative Error | Worst Relative Error |
|---|---|---|---|
| Q1 | 0.000101–0.006531 | 1.8300% | 18.0627% |
| Q2 | 0.006531–0.016533 | 1.4172% | 11.2978% |
| Q3 | 0.016533–0.028375 | 1.2029% | 9.7500% |
| Q4 | 0.028375–0.039952 | 1.1235% | 10.4426% |

The reported quartile statistic is the average, across test surfaces, of the maximum absolute relative error among each surface's 88 output points.

## Worst-Case Prediction

- Test surface index: 844
- Output grid index: 10
- True implied volatility: 0.180106
- Predicted implied volatility: 0.147574
- Absolute relative error: 18.0627%

Associated Heston parameters:

- xi0: 0.000220
- rho: -0.160301
- sigma: 0.054909
- theta: 0.137681
- kappa: 2.639619

## Preliminary Interpretation

The mean maximum prediction error decreases monotonically across increasing initial-variance quartiles, suggesting that lower initial variance is associated with larger worst-point prediction errors.

However, these descriptive results do not establish causality, statistical significance, or no-arbitrage violations.

## Next Research Tasks

1. Identify the strike and maturity corresponding to output grid index 10.
2. Examine the full distribution of prediction errors, rather than maximum errors alone.
3. Convert predicted implied volatilities into option prices.
4. Implement and validate no-arbitrage diagnostics against reference prices.
5. Evaluate the relationship between prediction accuracy and arbitrage violations.