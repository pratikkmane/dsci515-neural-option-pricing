# Step 6: repairs and H3 (protocol, written before the main runs)

**Question (H3).** Does training on strike derivatives restore no-arbitrage consistency at a smaller accuracy cost than adding a direct arbitrage penalty?

## Set-up
- Model: pointwise price network (softplus, width 64, depth 3, 60 epochs, cosine schedule: the step 5 budget), the family that violated most in step 5.
- Training size 5,000 surfaces (not 10,000: the repairs need second-order autograd, about 5 times the cost of a plain run). Noise 0 and 2 percent, seeds 0 to 2 (3 seeds; more if time allows). Same training sets as every other experiment (nested Sobol, seed-specific).
- Derivative loss: `L0 + w s0 [RMSE((dC/dK - slope)/s1) + RMSE((d2C/dK2 - density)/s2)]` at the 88 grid points of each surface, with exact strike derivatives from the Heston integral (`H.call_strike_derivatives`, validated against finite differences). The derivative labels carry the same multiplicative noise as the prices (no oracle). One ablation uses the slope term only. Weights 0.1 and 1.
- Arbitrage penalty: one-sided hinge on dC/dK > 0, d2C/dK2 < 0 and dC/dT < 0 at random continuous (strike, maturity) points in the box, with the batch's parameters (no labels; label scales only normalise it). Weights 1, 10, 100, 1000.
- Control: the same network with squared-error loss only (same code path, same seeds).
- Weights were chosen from a validation-set pilot (`results/pilot_repairs.json`, 2,000 surfaces, one seed, validation data only, no test data): the derivative loss at weights 0.01 to 10 and the penalty at 0.01 to 100 gave, respectively, no clear reduction (dense-grid violating cells 5.7 to 7.0 percent against 7.5 for the control) and a reduction that grew with the weight (6.2 percent at 0.01, 0.15 percent at 100, with a worse accuracy at 10 and above). The penalty range is extended to 1000 because 100 had not removed all violating surfaces; the derivative weight 10 is dropped because it was worse on both counts in the pilot.

## Measures (fixed now)
- Violation rate: largest of the three per-condition cell rates on the dense grid (101 strikes x 20 maturities) for the first 500 test surfaces. Target: below 1 percent. The base-grid surface rate on the 20,000 test surfaces is also reported.
- Accuracy: implied vol RMSE and price RMSE on the 20,000 test surfaces. Accuracy cost: paired difference to the control with the same seed and noise.
- Shape: autodiff strike slope and density error against the exact labels on the base grid (500 surfaces).
- Verdict rule for H3: supported if, at the cheapest weight of each family that reaches the target, the derivative loss has a smaller accuracy cost than the penalty with the paired bootstrap interval of the difference below zero; refuted if no derivative weight reaches the target while a penalty weight does (then the answer is "the penalty is the only one that repairs it"). Intervals over 3 seeds are wide and will be stated as such.

## Limits to carry into the write-up
- Derivative labels are only given at the 88 grid points, so a pointwise network can fit them there and still violate between them; the penalty is applied at random points, which is an advantage of that method rather than of its loss.
- One architecture, one size, 3 seeds. Not corrected for multiple comparisons.
