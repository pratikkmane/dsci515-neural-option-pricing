# Accurate but Inconsistent: When Do Neural Option Pricers Break No-Arbitrage Rules?

DSCI-515 Statistical Learning course project (Option 1). We study whether neural option-pricing surrogates that are
accurate on average still violate static no-arbitrage conditions (Carr and Madan, 2005), and why.

Base paper: Horvath, Muguruza and Tomas (2021), *Deep learning volatility*, Quantitative Finance 21(1), 11-27
([arXiv:1901.09647](https://arxiv.org/abs/1901.09647)). Base code: [amuguruza/NN-StochVol-Calibrations](https://github.com/amuguruza/NN-StochVol-Calibrations) (MIT).

## Layout
- `src/` library and reproduction code
- `scripts/` experiment runners
- `notebooks/` analysis notebooks
- `data/` Heston training set from the base repo (12,000 samples: 5 parameters + 88 implied vols)
- `results/` metrics and figures
- `app/` demo (planned)
- `docs/` proposal and notes

## Reproduction (step 1)
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/reproduce_heston.py --data data/HestonTrainSet.txt.gz --epochs 200
```
See `results/reproduction.json` for the latest numbers.

## Team workflow
- Work on a branch, open a pull request, get one review, then merge into `main`.
- One owner per workstream: reproduction and data, baselines and manipulations, repairs and analysis, demo.

## Credits
Data and base architecture from the authors' MIT-licensed repository, cited above.
