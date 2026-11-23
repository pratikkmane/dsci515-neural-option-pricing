# CLAUDE.md: DSCI-515 project "Accurate but Inconsistent"
Project title (no question mark): Accurate but Inconsistent: When Neural Option Pricers Break No-Arbitrage Rules

Read this first. It states what the project is, what the course requires, what is done, and what to do next.
Style rules: British English. No em dashes. Never invent citations, results or numbers; mark anything unverified.
Work on a branch, keep commits small, open pull requests (team of three).

## 1. The project in one paragraph
Neural networks can price options and calibrate volatility models in milliseconds. We study whether such a
surrogate, accurate on average, still violates the static no-arbitrage rules every valid option price must obey,
and why. Ground truth comes from the exact Heston (1993) model, so we can measure failure without human labels.
This is a statistical learning course project (DSCI-515), Option 1: a task models already do well, where we
investigate how and when it fails.

## 2. Course requirements (what the grade depends on)
- Research-driven: gain insight into why a model behaves as it does. Do not just implement an algorithm.
- Prefer: reproduce a paper with open code, then find insights the paper did not explore.
- Option 1 rules: manipulate inputs meaningfully (noise, size, representation, loss). Severe out-of-distribution
  tests alone are not enough. If no failure is found, report the attempts and explain, with references, why the
  model stayed robust. Honest negative or surprising results are fine.
- An "interesting insight" is unobserved, surprising or counterintuitive (example: retrieval models exploiting
  lexical overlap between query and function name; the example paper's refuted hypothesis counted as a result).
- Proposal: exactly four sections (abstract, introduction, motivation, experimental setting), at most 4 pages.
  The proposal already exists (see section 9).
- A presentation discussing relevant papers is also required. Final deliverable format, deadlines, and whether a
  live demo fits the slot are NOT yet confirmed with the instructor. Do not assume them.
- Evidence must come from experiments, with seeds and confidence intervals. The demo only illustrates findings.

## 3. Research question and hypotheses
Question: when and why does a neural option pricer that is accurate on average violate no-arbitrage conditions?
- H1 (accuracy hides shape): violations concentrate where strike-curvature error is large; curvature error
  predicts violating cells better (higher AUC) than pointwise error does.
- H2 (variance, not only extrapolation): violations occur inside the training range and grow as training size
  falls or label noise rises.
- H3 (the loss decides): training on derivatives restores consistency at a smaller accuracy cost than adding a
  direct arbitrage penalty.
Why failure is plausible: a valid call surface is decreasing and convex in strike and increasing in maturity
(Carr and Madan, 2005). Convexity in strike is the implied density, so a violation means negative probability.
A squared-error loss scores price levels only, and a second derivative amplifies small level errors.

## 4. Experimental design
Data: exact Heston prices. Sample parameter sets with a Sobol sequence over the ranges below. Implied vols on the
base 88-point grid (11 strikes 0.5 to 1.5 by 8 maturities 0.1 to 2.0). Also call prices on a denser grid so
convexity can be checked reliably. Planned training sizes 2k, 10k, 50k, 250k. Test set 20k fresh parameter sets,
plus a boundary subset within 10 percent of the parameter bounds. The shipped base data has only 12,000 samples,
so larger sizes need our own Heston pricer (characteristic function with numerical integration).

Heston parameter ranges (from the base code): xi0 0.0001 to 0.04, rho -0.95 to -0.1, sigma 0.01 to 1.0,
theta 0.01 to 0.2, kappa 1 to 10.

Models: base-paper network and variants; ridge on polynomial features (degree by cross-validation); gradient
boosting and a Gaussian process on leading principal components of the surface.

Manipulations (one factor at a time, plus a small crossed design for size and noise):
| Factor | Levels | Purpose |
| --- | --- | --- |
| Training size | 2k, 10k, 50k, 250k | estimation variance (H2) |
| Label noise | 0, 0.1, 0.5, 2 percent, multiplicative | Monte Carlo or market noise (H2) |
| Representation | grid output (base paper) or pointwise (params, strike, maturity) | expose curvature via autodiff (H1) |
| Capacity, activation | depth 2 to 6, width 30 to 256; ELU, softplus, ReLU | ReLU pointwise nets have zero curvature a.e. (H1) |
| Loss | squared error; plus derivative loss; plus arbitrage penalty | compare repairs (H3) |
| Input region | interior or near bounds | separate extrapolation from variance (H2) |
| Seeds | 10 per setting | variance and CIs |

Metrics:
- Accuracy: RMSE and 99th percentile error of implied vol.
- Shape: relative error of strike curvature, maturity slope, and Jacobian with respect to parameters.
- Consistency: share of surfaces and grid cells violating each Carr and Madan condition; negative probability mass.
- Statistics: 95 percent bootstrap CIs; bias-variance decomposition from bootstrap-trained models; AUC of local
  error as a predictor of violations.
Run the violation detector on exact Heston surfaces first to set a tolerance with a near-zero false-positive rate.

Hypothesis tests:
- H1 supported if curvature-error AUC exceeds pointwise-error AUC with the bootstrap interval of the difference above zero.
- H2 supported if interior violation rate is clearly above the detector false-positive rate and rises as data
  shrink and noise grows, with the variance term tracking it; refuted if violations appear only near the bounds.
- H3 supported if derivative supervision reaches a target violation rate (for example under 1 percent of cells)
  with a smaller accuracy loss than the penalty, shown as an accuracy against violation frontier.

## 5. Base paper and code
- Paper: Horvath, Muguruza, Tomas (2021), Deep learning volatility, Quantitative Finance 21(1), 11-27;
  arXiv:1901.09647.
- Code: github.com/amuguruza/NN-StochVol-Calibrations (MIT). Heston notebook: Heston/NNHeston.ipynb.
  The original notebook is Keras 2 era and Windows-specific (path "..\Data", old Keras import), so it is not
  runnable as is. We use a PyTorch port instead (src/reproduce_heston.py).
- Network: 5 inputs (parameters), three hidden layers width 30 with ELU, linear output of 88 values; RMSE loss;
  Adam; batch 32; 200 epochs; 85/15 split with random_state 42; surface standardised, parameters scaled to [-1, 1].
- Data: data/HestonTrainSet.txt.gz, 12,000 rows = 5 parameters then 88 implied vols.

## 6. Current status
Done:
- Four-section proposal submitted (docs/DSCI-515_Project_Proposal_final.pdf, 4 pages). Initial presentation given.
- Heston surrogate reproduced in PyTorch. Result on 1,800 test surfaces after 200 epochs (77 s on CPU):
  mean relative error 0.18 percent, RMSE 0.0009 implied vol (about 0.09 vol points), worst case 18 percent.
  Saved in results/reproduction.json (key rmse_iv, raw implied-vol units). A later re-run matched to about 14 digits.
- Compared with the authors' own notebook log: final validation RMSE 0.0279 (ours) against 0.0298 (theirs) in
  standardised units, one seed; epoch-to-epoch variation is about 0.002, and the two definitions differ slightly,
  so call it consistent, not matching (docs/reproduction_comparison.md).
- Worst case located: test surface 844, maturity 0.1, strike 1.5 (18.06 percent). Errors are largest at short
  maturities and for low initial variance. Descriptive only, one seed. Found independently in
  results/reproduction_analysis.md and docs/reproduction_comparison.md.
- Exact Heston pricer: src/heston.py (vectorised, Sobol sampler, implied vol inversion, dense grid), validated in
  docs/heston_pricer.md. Matches a 30-digit reference to about 3e-13 and the shipped data to about 1e-12 at maturities
  0.6, 1.2, 1.8, 2.0. Conventions confirmed: spot 1, zero rates, theta is long-run variance, 88 values maturity-major.
- At maturities 0.1, 0.3, 0.9, 1.5 the shipped vols are slightly lower than exact (RMS 5.9e-4 at 0.1); cause unknown,
  simple interpolation between maturities does not explain it. About 1.9 percent of 88-grid cells (mostly maturity
  0.1, strikes 0.5 to 0.7) have out-of-the-money prices below 1e-11 and give NaN: mask them, do not fill.
- Violation detector: src/violations.py (call spread, butterfly and calendar margins in price units, per-point flags,
  negative probability mass), documented in docs/violations.md. Tolerance 1e-12, set from exact surfaces (largest
  margin 2.3e-14, 32 parameter corners included); zero false positives in 31.7 million checks on fresh exact surfaces.
  Sensitivity depends on the grid (finer strike step is more sensitive) and is lowest-headroom at short maturities and in
  the wings, so always compare models on the same fixed grid and report dense and base grids separately.
- Step 4 code: src/data.py (nested Sobol training sets, fixed 20k test set with interior and boundary labels, label noise),
  src/evaluate.py, src/baselines.py (ridge, gradient boosting, GP; GP capped at 3,000 points), src/networks.py (grid network
  and pointwise network with autodiff strike curvature), runners in scripts/. Documented in docs/models.md. Preliminary runs
  only (one seed, 2,000 and 10,000 training surfaces, noise 0, 0.5, 2 percent): they show the pipeline works and a hint that
  accuracy and violation rates can move apart under label noise. NOT evidence for any hypothesis yet.
- Step 5 experiments: 482 runs (10 seeds for 2k to 50k, 5 for 250k, architecture and pointwise, 3 for the GP), pipeline in
  src/experiment.py, scripts/run_experiments.py and scripts/analyse_experiments.py; protocol written before the results, in
  docs/experiments.md (findings and limits there; generated tables in docs/experiments_results.md and results/tables/). Main
  results, all descriptive and with caveats: (1) accuracy and violation rate are only loosely tied; (2) pointwise networks that
  output the call price have a tiny price RMSE (0.3e-3 to 1.3e-3) but violate on 92 to 99 percent of surfaces, while vol-output
  ones violate on 0.04 to 0.13 percent; (3) label noise raises violations far more than RMSE (grid network, 2k surfaces: 0.3 to
  4.6 percent) and the effect vanishes with data (250k: no change); (4) ridge, boosting and GP keep about 1 to 2 percent
  violating at any size (bias, not variance); (5) H2 holds for the base network only with noisy labels and small data (interior
  violations 0.9 percent at 2k, none at 50k or 250k; none at all with clean labels); (6) wide networks with noisy labels violate
  much more (6 to 9 percent against 1.5); (7) H1 is true by construction and informative only under noise. H3 not tested yet.
- Step 6 repairs: 48 runs (pointwise price net, 5,000 surfaces, 3 seeds, noise 0 and 2 percent; protocol docs/repairs.md, tables docs/repairs_results.md). No derivative-loss weight reached under 1 percent violating cells (6 percent against 9 for the control) although the density error fell from 0.30 to 0.07; the arbitrage penalty did (0.6 percent at weight 10) with a lower implied vol RMSE than the control and a slightly higher price RMSE. So H3 is not supported in this set-up, with the caveat that the penalty uses random points and the derivative loss only grid points. Descriptive, 3 seeds.
- Step 7 demo: Gradio app in app/ (exact surface against grid network, ridge, price network and price network with the arbitrage penalty; density, violation maps, table, search for a breaking setting), numpy-only at run time, tested from the built bundle; not yet deployed (docs/demo.md).
Not done / not verified:
- Step 6 follow-ups: derivative loss with labels at random points; more seeds; other sizes. Experiment parts are reduced in seeds and levels (see
  docs/experiments.md, deviations); the 250k size has 5 seeds. Nothing is corrected for multiple comparisons.
- The experiment grid (all sizes, noise levels, 10 seeds, activations, depths) has not been run; the 250k size is a stretch.
  Pointwise networks need a training budget set before they can be compared with grid networks (docs/models.md, decisions).
- NOT yet compared with the paper's reported Heston errors (arxiv.org is unreachable from the cloud environment;
  someone with access should read the Heston section and add the figures to docs/reproduction_comparison.md).
- src/heston_reference.py (scalar pricer) disagrees with src/heston.py away from the money at maturities of 0.6 and
  above (median price difference up to 5e-3; scripts/compare_reference_pricer.py). src/heston.py is the one that matches
  the shipped data. Cause not found. results/heston_grid_validation.md and results/reproduction_analysis.md used the
  reference pricer for price comparisons, so read their price and implied-vol differences with this in mind.
- The authors' notebook was not run as published (old Keras import, Windows path). Do not claim it fails; say it was
  written for 2019-era Keras and we ported it.
- The claim that no controlled study of this exact question exists rests only on papers found so far.
- Presentation feedback (85/100): the research question appeared late (slide 12 of 17) and in several wordings.
  In the next presentation state it once, early, in one wording.

## 7. Next tasks, in order
1. Partly done: compared with the authors' notebook log. Still to do: the paper's own reported Heston errors.
2. DONE: src/heston.py (see docs/heston_pricer.md). Open: find why src/heston_reference.py disagrees with it.
3. DONE: src/violations.py (see docs/violations.md). Evaluate networks in float64; the implied-vol route cannot check
   the 1.9 percent of base-grid cells with unresolved wing prices.
4. DONE as code (see docs/models.md); preliminary runs only. Open: pointwise output (price or vol), training budget.
5. DONE (see docs/experiments.md). Re-run: python scripts/run_experiments.py <plan>, then python scripts/analyse_experiments.py.
6. DONE (see docs/repairs.md): derivative loss and arbitrage penalty; frontier in results/repairs/frontier.png.
7. Demo built (see docs/demo.md; needs deploying as a Hugging Face Space by an account owner). Gradio app. Sliders for parameters; exact, network and baseline surfaces side by side; violating
   regions highlighted. Deploy as a Hugging Face Space (free tier is CPU only and sleeps when idle).
8. Write-up and presentation. Papers to discuss: the base paper plus the references below.

## 8. Verified references (existence, authors, venue checked; full texts not all read)
1. Ackerer, Tagasovska, Vatter (2020). Deep smoothing of the implied volatility surface. NeurIPS 2020. arXiv:1906.05065.
2. Carr, Madan (2005). A note on sufficient conditions for no arbitrage. Finance Research Letters 2(3), 125-130.
3. Heston (1993). A closed-form solution for options with stochastic volatility with applications to bond and
   currency options. Review of Financial Studies 6(2), 327-343.
4. Horvath, Muguruza, Tomas (2021). Deep learning volatility. Quantitative Finance 21(1), 11-27. arXiv:1901.09647.
5. Huge, Savine (2020). Differential machine learning. arXiv:2005.02347 (preprint).
6. Itkin (2019). Deep learning calibration of option pricing models: some pitfalls and solutions. arXiv:1906.03507 (preprint).
7. Ruf, Wang (2020). Neural networks for option pricing and hedging: a literature review. Journal of
   Computational Finance. DOI 10.21314/JCF.2020.390.

## 9. Repository layout and commands
- src/ library code (heston.py is the pricer to use; heston_reference.py is a scalar cross-check); scripts/ experiment
  runners and validation; tests/ fast checks (test_heston.py, test_violations.py, test_baselines.py, test_networks.py, test_analysis.py); notebooks/ analysis; data/ Heston training
  set; results/ metrics and figures; app/ demo; docs/ proposal and notes.
- The submitted proposal is docs/DSCI-515_Project_Proposal_final.pdf.
- Setup: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
- Reproduce: python src/reproduce_heston.py --data data/HestonTrainSet.txt.gz --epochs 200
- Training is small (minutes on CPU). Use a GPU only for the largest runs. Fix seeds and log every run to results/.

## 10. Rules for working here
- Report failures and null results honestly; the course rewards explaining behaviour, not good numbers.
- Do not change the evaluation metrics after seeing results without writing the change in docs/.
- Keep manipulations mostly inside the training distribution; do not rely on extreme extrapolation tests.
- Ask before adding heavy dependencies or committing large files (data files over 50 MB belong outside git).
