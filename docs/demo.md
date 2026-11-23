# Class demo (step 7)

A Gradio app (`app/`) that illustrates the findings. It is not evidence: the models are single examples (seed 0, clean labels). The evidence is in `docs/experiments.md` and `docs/repairs.md`.

## What it shows
- Five sliders for the Heston parameters, a maturity selector, three presets and a "try to break a network" button that searches 300 random settings for the worst case of a chosen network.
- Exact surface (Heston integral) against four models: the base-paper grid network, polynomial ridge, a price-output pointwise network (squared error) and the same network with the arbitrage penalty at weight 10.
- Implied density against strike at the chosen maturity (negative parts are negative probability), maps of the violating points over strike and maturity, and a table of implied vol and price error and the violating share of checks per condition.
- Price networks are checked on the dense grid (101 strikes, 20 maturities); the grid network and ridge only on the 11 x 8 grid they output. The two are not directly comparable and the app says so.

## Files
- `app/core.py` (logic, no UI), `app/app.py` (UI), `app/surrogates.py` (numpy forward pass: no torch needed at demo time), `app/models/*.npz` (trained weights, under 1 MB in total).
- `scripts/train_demo_models.py` retrains the four models (about 10 minutes on CPU); `scripts/build_space.py` assembles the upload folder `dist/space/`.
- `tests/test_app.py`: the numpy forward pass equals the torch network; one evaluation, one figure pair and one search run end to end.

## Run locally
    pip install -r app/requirements.txt
    python app/app.py

## Deploy on Hugging Face Spaces (free CPU tier; the Space sleeps when idle and wakes on the first visit)
1. Create a free account at huggingface.co and a new Space: SDK Gradio, public (hardware: CPU basic).
2. `python scripts/build_space.py`, then upload the contents of `dist/space/` to the Space (web "Files" tab, or `git clone` the Space and copy them in, then push).
3. The Space builds in a few minutes. Add the link to the README and the slides.

## Limits
- Whether a live demo fits the presentation slot is not confirmed with the instructor; the app also works as screenshots.
- One seed per network: a different seed would break in different places. The "worst of 300" figure is for this network only.
