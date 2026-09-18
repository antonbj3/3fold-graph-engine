#!/usr/bin/env python3
"""E7: MEASURED properties of a local typed judge (Qwen2.5-0.5B-Instruct, CPU). Mechanism = the one in the
seed: the state is read once, every typed question is answered from the logits over its allowed options.
TEST-TEXT with exact labels: sentences that ASSERT a sign between two quantities, 8 templates with inversions
and negation, both signs for every pair — so half of the sentences contradict textbook physics. The label is
what the sentence asserts. 4 phrasings of the question × 2 option orders = 8 "lenses" per sentence.
Measured: determinism, accuracy, calibration (ECE), error correlation between lenses (rho), N_eff, whether
order-swapped lenses are the opposing channel, and naive vs tempered pooling of the 8 answers."""
import json, sys, time, os
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(6); torch.manual_seed(0)
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")     # a local directory or a Hugging Face id
DEV = "cuda" if os.environ.get("E7_DEVICE") == "cuda" and torch.cuda.is_available() else "cpu"; TAG = os.environ.get("JUDGE_TAG", "")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left"); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()

PAIRS = [("temperature", "electrical resistance of copper", 1), ("pressure", "boiling point of water", 1), ("altitude", "air pressure", -1),
         ("grain size", "yield strength of steel", -1), ("cooling rate", "grain size", -1), ("porosity", "fatigue strength", -1),
         ("temperature", "viscosity of oil", -1), ("wire length", "electrical resistance", 1), ("wire cross-section", "electrical resistance", -1),
         ("spring stiffness", "oscillation frequency", 1), ("mass", "oscillation frequency", -1), ("voltage", "current through a resistor", 1),
         ("distance", "gravitational force", -1), ("carbon content", "hardness of steel", 1), ("temperature", "solubility of sugar in water", 1),
         ("frequency", "photon energy", 1), ("wavelength", "photon energy", -1), ("concentration", "reaction rate", 1),
         ("surface roughness", "friction", 1), ("insulation thickness", "heat loss", -1)]
def sentences(x, y, s):
    up, dn = ("an increase", "a decrease") if s > 0 else ("a decrease", "an increase")
    return [f"An increase in {x} leads to {up} in {y}.", f"The {y} {'rises' if s > 0 else 'falls'} as {x} grows.",
            f"Reducing {x} {'lowers' if s > 0 else 'raises'} the {y}.", f"Higher {x} is associated with {'higher' if s > 0 else 'lower'} {y}.",
            f"The {y} is {'directly' if s > 0 else 'inversely'} proportional to {x}.", f"When {x} drops, the {y} {'drops' if s > 0 else 'climbs'}.",
            f"It is not the case that the {y} {'decreases' if s > 0 else 'increases'} with {x}; it {'increases' if s > 0 else 'decreases'}.",
            f"Samples with lower {x} showed {'lower' if s > 0 else 'higher'} {y}."]
QUESTIONS = ["According to the text, when {x} increases, the {y}:", "Per the statement above, raising {x} makes the {y}:",
             "The text claims that with more {x}, the {y}:", "Based only on the sentence, if {x} goes up, the {y}:"]
items = [(t, x, y, s, s == phys, k) for x, y, phys in PAIRS for s in (1, -1) for k, t in enumerate(sentences(x, y, s))]

def prompt(text, q, x, y, order):
    a, b = ("increases", "decreases") if order == 0 else ("decreases", "increases")
    msg = f"Text: {text}\n\n{q.format(x=x, y=y)}\nA) {a}\nB) {b}\nAnswer with A or B."
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
A, B = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
@torch.no_grad()
def p_increase(prompts, orders, bs=32):
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEV); lg = model(**enc).logits[:, -1, :].float()
        pa = torch.softmax(lg[:, [A, B]], -1)[:, 0].cpu().numpy(); out.extend(pa)
    pa = np.array(out); return np.where(np.array(orders) == 0, pa, 1 - pa)       # P("increases")

def main():
    lenses = [(qi, o) for qi in range(4) for o in (0, 1)]
    P = np.zeros((len(items), len(lenses))); t0 = time.time()
    for li, (qi, o) in enumerate(lenses):
        P[:, li] = p_increase([prompt(t, QUESTIONS[qi], x, y, o) for t, x, y, *_ in items], [o] * len(items))
    elapsed = time.time() - t0
    again = p_increase([prompt(t, QUESTIONS[0], x, y, 0) for t, x, y, *_ in items[:64]], [0] * 64)
    truth = np.array([s > 0 for _, _, _, s, _, _ in items]); agree_phys = np.array([a for *_, a, _ in items])
    err = (P > 0.5) != truth[:, None]
    def ece(p, t, bins=10):
        conf = np.maximum(p, 1 - p); ok = (p > 0.5) == t; e = 0.0
        for lo in np.linspace(0.5, 1, bins + 1)[:-1]:
            m = (conf >= lo) & (conf < lo + 0.5 / bins + 1e-9)
            if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
        return float(e)
    C = np.corrcoef(err.T.astype(float)); iu = np.triu_indices(len(lenses), 1)
    same_order = np.array([lenses[a][1] == lenses[b][1] for a, b in zip(*iu)])
    rho_all, rho_same, rho_swap = float(np.nanmean(C[iu])), float(np.nanmean(C[iu][same_order])), float(np.nanmean(C[iu][~same_order]))
    K = len(lenses); neff = K / (1 + (K - 1) * max(rho_all, 0))
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    pool = lambda w: 1 / (1 + np.exp(-w * logit(P).sum(1)))
    res = {"model": (MODEL if not os.path.isdir(MODEL) else Path(MODEL).name) + f" ({DEV})", "n_sentences": len(items), "lenses": K, "seconds_for_all": round(elapsed, 1),
           "judgments_per_second": round(len(items) * K / elapsed, 1),
           "deterministic_max_abs_diff_on_repeat": float(np.abs(again - P[:64, 0]).max()),
           "accuracy_per_lens": [round(float(1 - e), 3) for e in err.mean(0)], "ece_per_lens_mean": round(float(np.mean([ece(P[:, k], truth) for k in range(K)])), 3),
           "accuracy_when_text_agrees_with_physics": round(float(1 - err[agree_phys].mean()), 3),
           "accuracy_when_text_contradicts_physics": round(float(1 - err[~agree_phys].mean()), 3),
           "accuracy_by_template": [round(float(1 - err[np.array([k for *_, k in items]) == j].mean()), 3) for j in range(8)],
           "error_correlation_mean": round(rho_all, 3), "error_correlation_same_option_order": round(rho_same, 3),
           "error_correlation_swapped_option_order": round(rho_swap, 3), "n_eff_of_8_lenses": round(neff, 2),
           "pooled_naive": {"accuracy": round(float(((pool(1.0) > 0.5) == truth).mean()), 3), "ece": round(ece(pool(1.0), truth), 3)},
           "pooled_tempered_by_n_eff": {"accuracy": round(float(((pool(neff / K) > 0.5) == truth).mean()), 3), "ece": round(ece(pool(neff / K), truth), 3)},
           "pooled_one_swapped_pair": {"accuracy": round(float((((logit(P[:, 0]) + logit(P[:, 1])) > 0) == truth).mean()), 3)},
           "pooled_two_same_order": {"accuracy": round(float((((logit(P[:, 0]) + logit(P[:, 2])) > 0) == truth).mean()), 3)}}
    np.save(Path(__file__).parent / f"e7_probabilities{TAG}.npy", P.astype(np.float32))
    json.dump(res, open(Path(__file__).parent / f"e7_results{TAG}.json", "w"), indent=1); [print(k, v) for k, v in res.items()]


if __name__ == "__main__":
    main()
