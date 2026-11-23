"""Gradio demo for 'Accurate but Inconsistent': move the Heston parameters, compare the exact surface with trained surrogates, and see
where they break the no-arbitrage rules. The demo illustrates the findings in docs/; the evidence is in the experiments, not here."""
import numpy as np
import gradio as gr

import core
import heston as H

MODELS = core.get_models()
LB, UB = H.LB, H.UB
INTRO = """# Accurate but inconsistent
Neural networks can price options in microseconds and are usually accurate. A valid call price must be **decreasing and convex in strike**
(convexity is the probability density, so a violation means negative probability) and **increasing in maturity**. Here the true prices come from
the exact Heston model, so every violation is a real failure. Move the sliders and see where each model breaks a rule.

*These are single example networks (one seed, trained on clean data). The study's evidence is in the repository's experiments: a price-output network
has a small price error (about 0.001) yet violates the rules on most surfaces, and adding an arbitrage penalty removes most of the violations.*"""
NOTES = """In the maps, dark red marks points that take part in a violated check; pale orange marks cells where the exact price is too small to check (those are skipped, as in the experiments). The base-paper network and ridge curves coincide here because both are close to the exact curve.

Reading the table: a *check* is a stencil of three neighbouring strikes (convexity), two strikes (monotonicity) or two maturities (calendar).
Price networks are checked on a dense grid (101 strikes, 20 maturities), the base-paper network and ridge only on the 11 x 8 grid they output.
The tolerance is 1e-12 in price, calibrated so that the exact model never trips it."""


def run(xi0, rho, sigma, theta, kappa, mat):
    p = np.array([xi0, rho, sigma, theta, kappa], float)
    res = core.evaluate(p, MODELS)
    return core.fig_density(p, res, int(mat)), core.fig_map(res), core.table(res)


def attack(model_name):
    name = {v[0]: k for k, v in MODELS.items()}[model_name]
    p, worst, mean = core.search_breaking(MODELS[name][1])
    msg = f"Worst of 300 random settings for this model: {100 * worst:.0f}% of checks fail (average over the 300: {100 * mean:.1f}%)."
    return (*[float(x) for x in p], msg)


def preset(name):
    return [float(x) for x in core.PRESETS[name]]


with gr.Blocks(title="Accurate but inconsistent") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            sl = [gr.Slider(float(LB[i]), float(UB[i]), value=float(core.PRESETS["Typical"][i]), label=core.PARAM_LABELS[i]) for i in range(5)]
            mat = gr.Slider(0, len(H.MATURITIES) - 1, value=0, step=1, label="Maturity index (0 = 0.1 years, 7 = 2 years)")
            pre = gr.Dropdown(list(core.PRESETS), value="Typical", label="Presets")
            gr.Markdown("**Try to break a network**")
            target = gr.Dropdown([v[0] for v in MODELS.values() if v[1].kind == "pointwise"] or [v[0] for v in MODELS.values()],
                                 value=MODELS["price"][0] if "price" in MODELS else None, label="Network to attack")
            btn = gr.Button("Search 300 random settings for the worst case")
            msg = gr.Markdown()
        with gr.Column(scale=3):
            fig1 = gr.Plot(label="Implied density")
            fig2 = gr.Plot(label="Where the rules break")
    tab = gr.Dataframe(headers=core.TABLE_HEADER, interactive=False, wrap=True)
    gr.Markdown(NOTES)
    inputs = sl + [mat]
    outputs = [fig1, fig2, tab]
    for c in inputs:
        c.release(run, inputs, outputs) if hasattr(c, "release") else c.change(run, inputs, outputs)
    pre.change(preset, pre, sl).then(run, inputs, outputs)
    btn.click(attack, target, sl + [msg]).then(run, inputs, outputs)
    demo.load(run, inputs, outputs)

if __name__ == "__main__":
    demo.launch()
