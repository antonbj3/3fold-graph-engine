#!/usr/bin/env python3
"""E22: does the representation probe TRANSFER to text written by other people? e20 measured the probe on the author's own
320 templated sentences (trained on the symbolic rule's labels, tested on held-out quantity pairs). Here the test set is
QuaRTz (Tafjord et al. 2019, CC BY 4.0): each item is a science paragraph with two annotated property phrases and their
two directions; as in e15 the pair is (cause_prop, effect_prop) and the truth is direction(cause) x direction(effect),
one item per para_id.

Four readings on the SAME QuaRTz test paragraphs, per model size:
  (a) probe trained on OUR 320 templated sentences with the RULE's labels (polarity_rules, no human labels anywhere)
  (b) probe trained on the QuaRTz TRAIN split with its human labels
  (c) the model's token output (e7 prompt, lens 0) — the collapse to A/B
  (d) the symbolic rule itself (coverage = it answers, accuracy where it answers)
The layer of (a) and (b) is chosen on training data only (inner split over groups: quantity pairs for (a), property pairs
for (b)); the QuaRTz test split is touched once, at prediction.

Run per model: JUDGE_MODEL=<dir> JUDGE_TAG=_1p5b E7_DEVICE=cuda python3 e22_probe_lens_transfer.py
(each run writes e22_results<TAG>.json and re-merges every such file into e22_results.json + prints the table).
"""
import contextlib, glob, importlib, io, json, os, sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[3] / "models" / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"] + "/datasets")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    e7 = importlib.import_module("e7_typed_judge_measured")
from graph_engine.polarity_rules import asserted_sign
from graph_engine.representation_probe import RepresentationProbe
from datasets import load_dataset

HERE = Path(__file__).parent
TAG = e7.TAG or "_0p5b"
NAME = Path(e7.MODEL).name


def quartz_rows():
    """e15's construction, verbatim: paragraph + the two annotated property phrases, truth = product of the two MORE/LESS signs."""
    ds = load_dataset("allenai/quartz")
    rows, seen = [], set()
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            a = ex["para_anno"]
            a = eval(a) if isinstance(a, str) else a
            cp, ep = a.get("cause_prop", "").strip(), a.get("effect_prop", "").strip()
            cs, es = a.get("cause_dir_sign", ""), a.get("effect_dir_sign", "")
            if not (cp and ep and cs in ("MORE", "LESS") and es in ("MORE", "LESS")):
                continue
            if ex["para_id"] in seen:
                continue
            seen.add(ex["para_id"])
            rows.append((ex["para"], cp, ep, (1 if cs == "MORE" else -1) * (1 if es == "MORE" else -1), split))
    return rows


rows = quartz_rows()
qtexts = [r[0] for r in rows]; qpairs = [(r[1], r[2]) for r in rows]
qtruth = np.array([r[3] for r in rows]) > 0; qsplit = np.array([r[4] for r in rows])
tr, te = qsplit == "train", qsplit == "test"            # validation is left unused, as in e15's reporting
sub = lambda a, m: [v for v, k in zip(a, m) if k]

# ---- the model, reused from e7 (already loaded, right device/dtype) -------------------------------------------------
probe_kw = dict(device=e7.DEV, batch_size=8, name=NAME)
items = e7.items                                        # 320 templated sentences
otexts = [t for t, x, y, *_ in items]; opairs = [(x, y) for t, x, y, *_ in items]
orule = np.array([asserted_sign(t, x, y) for t, x, y, *_ in items]); assert (orule != 0).all()

# (a) trained on our templates with the rule's labels
pa = RepresentationProbe((e7.tok, e7.model), **probe_kw).fit(otexts, opairs, orule > 0)
acc_a = float(((pa.predict_proba(sub(qtexts, te), sub(qpairs, te)) > 0.5) == qtruth[te]).mean())
# (b) trained on the QuaRTz train split with human labels
pb = RepresentationProbe((e7.tok, e7.model), **probe_kw).fit(sub(qtexts, tr), sub(qpairs, tr), qtruth[tr])
proba_b = pb.predict_proba(sub(qtexts, te), sub(qpairs, te))
acc_b = float(((proba_b > 0.5) == qtruth[te]).mean())
# (c) the token output on the same paragraphs
prompts = [e7.prompt(t, e7.QUESTIONS[0], x, y, 0) for t, (x, y) in zip(sub(qtexts, te), sub(qpairs, te))]
p_out = e7.p_increase(prompts, [0] * int(te.sum()))
acc_c = float(((p_out > 0.5) == qtruth[te]).mean())
# (d) the rule on the same paragraphs
rule_te = np.array([asserted_sign(t, x, y) for t, (x, y) in zip(sub(qtexts, te), sub(qpairs, te))])
ans = rule_te != 0
acc_d = float(((rule_te > 0) == qtruth[te])[ans].mean())
proba_a = pa.predict_proba(sub(qtexts, te), sub(qpairs, te))

res = {
    "model": NAME, "size_tag": TAG.lstrip("_"), "device": e7.DEV, "layers": pa.n_layers,
    "n_quartz_paragraphs": {"train": int(tr.sum()), "test": int(te.sum()), "all_unique_para_id": len(rows)},
    "majority_class_on_test": round(float(max(qtruth[te].mean(), 1 - qtruth[te].mean())), 3),
    "a_probe_trained_on_our_templates_rule_labels": {"accuracy": round(acc_a, 3), "n": int(te.sum()),
                                                     "layer": pa.layer, "n_train": len(items),
                                                     "train_labels": "polarity_rules.asserted_sign"},
    "b_probe_trained_on_quartz_train_human_labels": {"accuracy": round(acc_b, 3), "n": int(te.sum()),
                                                     "layer": pb.layer, "n_train": int(tr.sum()),
                                                     "train_labels": "QuaRTz annotation"},
    "c_token_output_e7_prompt": {"accuracy": round(acc_c, 3), "n": int(te.sum())},
    "d_polarity_rule": {"coverage": round(float(ans.mean()), 3), "accuracy_where_answered": round(acc_d, 3),
                        "n_answered": int(ans.sum()), "n": int(te.sum())},
    "share_predicted_increase": {"truth": round(float(qtruth[te].mean()), 3), "a_probe": round(float((proba_a > 0.5).mean()), 3),
                                 "b_probe": round(float((proba_b > 0.5).mean()), 3), "c_token_output": round(float((p_out > 0.5).mean()), 3)},
    "on_items_the_rule_abstains_on": {"n": int((~ans).sum()),
                                      "a_probe": round(float(((proba_a > 0.5) == qtruth[te])[~ans].mean()), 3),
                                      "b_probe": round(float(((proba_b > 0.5) == qtruth[te])[~ans].mean()), 3),
                                      "c_token_output": round(float(((p_out > 0.5) == qtruth[te])[~ans].mean()), 3)},
    "note": "QuaRTz label = direction(cause_prop) x direction(effect_prop) from para_anno (e15's mapping); one item per "
            "para_id; validation split unused; layers for (a) and (b) chosen on their own training data only",
}
json.dump(res, open(HERE / f"e22_results{TAG}.json", "w"), indent=1)

# ---- merge every per-model file and print the table -----------------------------------------------------------------
order = {"_0p5b": 0, "_1p5b": 1, "_3b": 2}
files = sorted(glob.glob(str(HERE / "e22_results_*.json")), key=lambda f: order.get("_" + Path(f).stem.split("results_")[1], 9))
merged = {"experiment": "e22", "test_set": "QuaRTz test split (paragraphs by other people), one item per para_id",
          "by_model": [json.load(open(f)) for f in files]}
json.dump(merged, open(HERE / "e22_results.json", "w"), indent=1)
hdr = f"{'model':<24}{'(a) our-templates':>18}{'(b) quartz-train':>18}{'(c) token out':>15}{'(d) rule cov/acc':>20}{'n':>6}"
print(hdr); print("-" * len(hdr))
for r in merged["by_model"]:
    a, b, c, d = (r["a_probe_trained_on_our_templates_rule_labels"], r["b_probe_trained_on_quartz_train_human_labels"],
                  r["c_token_output_e7_prompt"], r["d_polarity_rule"])
    print(f"{r.get('size_tag', r['model']):<24}{a['accuracy']:>13.3f}(L{a['layer']:<2}){b['accuracy']:>13.3f}(L{b['layer']:<2})"
          f"{c['accuracy']:>15.3f}{d['coverage']:>11.2f}/{d['accuracy_where_answered']:<8.3f}{c['n']:>6}")
print("majority class on test:", merged["by_model"][0]["majority_class_on_test"] if merged["by_model"] else "-")
for k, v in res.items():
    print(k, v)
