#!/usr/bin/env python3
"""E7c: the 0.5B judge fails when the sign must be COMPOSED ("Reducing X lowers Y" → +). Ask two atomic typed
fields instead — is X described as going up or down? is Y? — and multiply the signs in code. No labels used.
Each field is asked in both option orders (the opposing lens measured in e7) and the two log-odds are added."""
import json, sys
from pathlib import Path
import numpy as np
import e7_typed_judge_measured as j          # reuses model, items, prompt plumbing (re-runs nothing heavy: guarded below)
tok, items = j.tok, j.items
def prompt(text, target, order):
    a, b = ("going up", "going down") if order == 0 else ("going down", "going up")
    msg = f"Text: {text}\n\nIn the text, the {target} is described as:\nA) {a}\nB) {b}\nAnswer with A or B."
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
LX = sum(lg(j.p_increase([prompt(t, x, o) for t, x, y, *_ in items], [o] * len(items))) for o in (0, 1))
LY = sum(lg(j.p_increase([prompt(t, y, o) for t, x, y, *_ in items], [o] * len(items))) for o in (0, 1))
truth = np.array([s > 0 for _, _, _, s, _, _ in items]); tm = np.array([k for *_, k in items])
pred = (LX > 0) == (LY > 0)                                  # same direction → sign +
px, py = 1 / (1 + np.exp(-LX)), 1 / (1 + np.exp(-LY)); p_plus = px * py + (1 - px) * (1 - py)
xdir = np.array([k not in (2, 5, 7) for *_, k in items])     # templates 2,5,7 describe X going DOWN
res = {"composed_sign_accuracy": round(float((pred == truth).mean()), 3),
       "composed_by_template": [round(float((pred == truth)[tm == k].mean()), 3) for k in range(8)],
       "field_X_direction_accuracy": round(float(((LX > 0) == xdir).mean()), 3),
       "field_X_by_template": [round(float(((LX > 0) == xdir)[tm == k].mean()), 3) for k in range(8)],
       "brier_composed": round(float(np.mean((p_plus - truth) ** 2)), 3)}
json.dump(res, open(Path(__file__).parent / "e7c_results.json", "w"), indent=1); [print(k, v) for k, v in res.items()]
