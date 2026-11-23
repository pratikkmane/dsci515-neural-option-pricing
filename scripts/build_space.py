"""Assemble the folder to upload to a Hugging Face Space: python scripts/build_space.py [out_dir] (default dist/space, git-ignored).

Copies app/*.py, the trained .npz models, requirements, and the two library files the app needs (src/heston.py, src/violations.py),
and writes the Space README with its metadata header. Then upload the folder (see docs/demo.md)."""
import os, shutil, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist", "space")
shutil.rmtree(out, ignore_errors=True)
os.makedirs(os.path.join(out, "src"))
os.makedirs(os.path.join(out, "models"))
for f in ("app.py", "core.py", "surrogates.py", "requirements.txt"):
    shutil.copy(os.path.join(ROOT, "app", f), out)
for f in ("heston.py", "violations.py"):
    shutil.copy(os.path.join(ROOT, "src", f), os.path.join(out, "src"))
for f in os.listdir(os.path.join(ROOT, "app", "models")):
    if f.endswith(".npz"):
        shutil.copy(os.path.join(ROOT, "app", "models", f), os.path.join(out, "models"))
open(os.path.join(out, "README.md"), "w").write("""---
title: Accurate but inconsistent
emoji: 📉
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Accurate but inconsistent: when neural option pricers break no-arbitrage rules

Class demo for DSCI-515. Move the Heston parameters and compare the exact option price surface with trained surrogates; see where each breaks
the no-arbitrage rules (decreasing and convex in strike, increasing in maturity). Single example networks: the evidence is in the project
repository (github.com/pratikkmane/dsci515-neural-option-pricing).
""")
print("built", os.path.abspath(out), sorted(os.listdir(out)))
