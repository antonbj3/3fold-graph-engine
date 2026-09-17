#!/usr/bin/env python3
"""E14: two REAL channels on the 320 labelled TEST sentences of e7 — the measured answers of a local 0.5B language model
(e7_probabilities.npy, 8 lenses) and the symbolic rule graph_engine.polarity_rules. No simulation.
Reported: accuracy of each, their error correlation, and what "accept only where they agree" is worth.
CAVEAT: the sentences were generated from 8 templates written by the same author as the rule's lexicon. The rule was written
from the lexicon and the clause logic, not fitted to the labels, but this is not an independent text sample."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from graph_engine.polarity_rules import asserted_sign
from graph_engine.lens_pooling import logit, StratifiedPlatt
PAIRS = [("temperature", "electrical resistance of copper"), ("pressure", "boiling point of water"), ("altitude", "air pressure"), ("grain size", "yield strength of steel"),
         ("cooling rate", "grain size"), ("porosity", "fatigue strength"), ("temperature", "viscosity of oil"), ("wire length", "electrical resistance"),
         ("wire cross-section", "electrical resistance"), ("spring stiffness", "oscillation frequency"), ("mass", "oscillation frequency"), ("voltage", "current through a resistor"),
         ("distance", "gravitational force"), ("carbon content", "hardness of steel"), ("temperature", "solubility of sugar in water"), ("frequency", "photon energy"),
         ("wavelength", "photon energy"), ("concentration", "reaction rate"), ("surface roughness", "friction"), ("insulation thickness", "heat loss")]
def sentences(x, y, s):
    up, dn = ("an increase", "a decrease") if s > 0 else ("a decrease", "an increase")
    return [f"An increase in {x} leads to {up} in {y}.", f"The {y} {'rises' if s > 0 else 'falls'} as {x} grows.", f"Reducing {x} {'lowers' if s > 0 else 'raises'} the {y}.",
            f"Higher {x} is associated with {'higher' if s > 0 else 'lower'} {y}.", f"The {y} is {'directly' if s > 0 else 'inversely'} proportional to {x}.",
            f"When {x} drops, the {y} {'drops' if s > 0 else 'climbs'}.", f"It is not the case that the {y} {'decreases' if s > 0 else 'increases'} with {x}; it {'increases' if s > 0 else 'decreases'}.",
            f"Samples with lower {x} showed {'lower' if s > 0 else 'higher'} {y}."]
items = [(t, x, y, s, k) for x, y in PAIRS for s in (1, -1) for k, t in enumerate(sentences(x, y, s))]
truth = np.array([s > 0 for *_, s, _ in items]); tmpl = np.array([k for *_, k in items]); pair = np.repeat(np.arange(20), 16)
rule = np.array([asserted_sign(t, x, y) for t, x, y, _, _ in items]); answered = rule != 0
P = np.load(Path(__file__).parent / "e7_probabilities.npy").astype(float); L = logit(P)
out = {"rule": {"coverage": round(float(answered.mean()), 3), "accuracy_where_it_answers": round(float(((rule > 0) == truth)[answered].mean()), 3),
                "accuracy_by_template": [round(float(((rule > 0) == truth)[(tmpl == k) & answered].mean()), 3) for k in range(8)]},
       "lm_raw_pooled_8_lenses": {"accuracy": round(float(((L.sum(1) > 0) == truth).mean()), 3),
                                  "accuracy_by_template": [round(float(((L.sum(1) > 0) == truth)[tmpl == k].mean()), 3) for k in range(8)]}}
lm = L.sum(1) > 0; e_lm, e_rule = (lm != truth)[answered], ((rule > 0) != truth)[answered]
out["error_correlation_lm_vs_rule"] = None if e_rule.std() == 0 or e_lm.std() == 0 else round(float(np.corrcoef(e_lm, e_rule)[0, 1]), 3)
agree = answered & (lm == (rule > 0))
out["accept_only_where_they_agree"] = {"coverage": round(float(agree.mean()), 3), "accuracy": round(float((lm == truth)[agree].mean()), 3),
                                       "lm_accuracy_where_they_disagree": round(float((lm == truth)[answered & ~agree].mean()), 3)}
# the rule as a free source of LABELS: calibrate the language model per sentence form on the rule's answers instead of on ground truth
accs = []
for seed in range(20):
    rng = np.random.default_rng(seed); tr = np.isin(pair, rng.choice(20, 10, replace=False)); te = ~tr
    cal = StratifiedPlatt().fit(P[tr & answered], rule[tr & answered] > 0, tmpl[tr & answered]); Lc = cal.transform(P[te], tmpl[te])
    accs.append(float(((Lc.sum(1) > 0) == truth[te]).mean()))
out["lm_calibrated_on_RULE_labels_no_ground_truth"] = {"accuracy_held_out_pairs": round(float(np.mean(accs)), 3), "sd": round(float(np.std(accs)), 3),
                                                       "for_reference_calibrated_on_true_labels_e7b": 0.927}
json.dump(out, open(Path(__file__).parent / "e14_results.json", "w"), indent=1); [print(k, v) for k, v in out.items()]
