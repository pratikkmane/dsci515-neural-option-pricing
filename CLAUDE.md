# CLAUDE.md: DSCI-515 project "Accurate but Inconsistent"

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
- Four-section proposal drafted (section 9).
- Heston surrogate reproduced in PyTorch. Result on 1,800 test surfaces after 200 epochs (77 s on CPU):
  mean relative error 0.18 percent, RMSE 0.0009 implied vol (about 0.09 vol points), worst case 18 percent.
  Saved in results/reproduction.json.
- Reproduction compared with the original notebook (docs/reproduction_comparison.md): final validation RMSE and the
  per-cell error pattern are consistent with the original. One seed only.
- Worst case (18 percent) located: a single cell at maturity 0.1, strike 1.5; errors concentrate at short maturity
  and near the lower parameter bounds. Descriptive only, to be tested over seeds.
Not done / not verified:
- The paper's own reported Heston numbers have NOT been checked (arxiv.org is unreachable from the work
  environment). Someone with access should read the Heston section and add the figures to the comparison note.
- The claim that no controlled study of this exact question exists rests only on papers found so far.
- Proposal page count not checked (target at most 4 pages).

## 7. Next tasks, in order
1. DONE except the paper's own numbers (see section 6). Next: step 2.
2. Write src/heston.py: exact call pricer (characteristic function, numerical integration), Sobol parameter
   sampler, implied vol inversion, dense price grid. Validate against the shipped data file.
3. Write src/violations.py: Carr and Madan checks (decreasing and convex in strike, increasing in maturity),
   negative probability mass, tolerance calibrated on exact surfaces.
4. Baselines (ridge, gradient boosting, Gaussian process) and a pointwise network with autodiff curvature.
5. Manipulation grid with 10 seeds, then bootstrap CIs, AUC analysis, bias-variance decomposition.
6. Repairs: derivative loss and arbitrage penalty; accuracy against violation frontier.
7. Demo (app/): Gradio app. Sliders for parameters; exact, network and baseline surfaces side by side; violating
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
- src/ library code; scripts/ experiment runners; notebooks/ analysis; data/ Heston training set;
  results/ metrics and figures; app/ demo; docs/ proposal and notes.
- The proposal is an editable document on claude.ai (title: Accurate but Inconsistent: When Do Neural Option
  Pricers Break No-Arbitrage Rules?). Export it to PDF or Markdown into docs/ so it lives in the repo.
- Setup: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
- Reproduce: python src/reproduce_heston.py --data data/HestonTrainSet.txt.gz --epochs 200
- Training is small (minutes on CPU). Use a GPU only for the largest runs. Fix seeds and log every run to results/.

## 10. Rules for working here
- Report failures and null results honestly; the course rewards explaining behaviour, not good numbers.
- Do not change the evaluation metrics after seeing results without writing the change in docs/.
- Keep manipulations mostly inside the training distribution; do not rely on extreme extrapolation tests.
- Ask before adding heavy dependencies or committing large files (data files over 50 MB belong outside git).
