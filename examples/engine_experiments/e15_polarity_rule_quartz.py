#!/usr/bin/env python3
"""E15: the symbolic polarity rule on text written by OTHER people — QuaRTz (Tafjord et al. 2019, CC BY 4.0, 3 864 items,
allenai/quartz on the Hugging Face hub). Each item has a paragraph with an annotated cause property, effect property and the two
directions (MORE/LESS). Task for the rule: given the paragraph and the two annotated property phrases, return the sign =
direction(cause) × direction(effect). Ground truth from the annotation. Coverage = rule answers (non-zero)."""
import json, os, re
from pathlib import Path
import numpy as np
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[3] / "models" / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub"); os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"] + "/datasets")
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.polarity_rules import asserted_sign
from datasets import load_dataset
ds = load_dataset("allenai/quartz")
rows = []
for split in ("train", "validation", "test"):
    for ex in ds[split]:
        ex = dict(ex, split=split)
        a = ex["para_anno"]; a = eval(a) if isinstance(a, str) else a; cp, ep = a.get("cause_prop", "").strip(), a.get("effect_prop", "").strip()
        cs, es = a.get("cause_dir_sign", ""), a.get("effect_dir_sign", "")
        if not (cp and ep and cs in ("MORE", "LESS") and es in ("MORE", "LESS")):
            continue
        truth = (1 if cs == "MORE" else -1) * (1 if es == "MORE" else -1)
        rows.append((ex["para_id"], ex["para"], cp, ep, truth, ex["split"]))
seen, uniq = set(), []
for r in rows:
    if r[0] not in seen: seen.add(r[0]); uniq.append(r)
pred = np.array([asserted_sign(p, cp, ep) for _, p, cp, ep, _, _ in uniq]); truth = np.array([t for *_, t, _ in uniq]); split = np.array([sp for *_, sp in uniq])
ans = pred != 0
both_named = np.array([bool(re.search(re.escape(cp.lower()), p.lower())) and bool(re.search(re.escape(ep.lower()), p.lower())) for _, p, cp, ep, _, _ in uniq])
out = {"n_paragraphs": len(uniq), "both_property_phrases_found_verbatim": round(float(both_named.mean()), 3),
       "coverage_all": round(float(ans.mean()), 3), "accuracy_where_answered": round(float((pred == truth)[ans].mean()), 3),
       "coverage_when_both_phrases_found": round(float(ans[both_named].mean()), 3), "accuracy_when_both_phrases_found": round(float((pred == truth)[ans & both_named].mean()), 3),
       "n_answered": int(ans.sum()), "majority_class_share": round(float(max((truth > 0).mean(), (truth < 0).mean())), 3)}
for sp in ("train", "test"):
    m = split == sp
    out[f"{sp}_split"] = {"n": int(m.sum()), "coverage": round(float(ans[m].mean()), 3), "accuracy_where_answered": round(float((pred == truth)[ans & m].mean()), 3)}
out["note"] = "the comparative lexicon was extended after a first run on all splits (0.884 where answered, coverage 0.363); the test split was not looked at when extending it"
wrong = [(p[:140], cp, ep, int(t)) for (_, p, cp, ep, t, sp), q in zip(uniq, pred) if q != 0 and q != t and sp == 'train'][:12]
out["examples_wrong"] = wrong
json.dump(out, open(Path(__file__).parent / "e15_results.json", "w"), indent=1, ensure_ascii=False); [print(k, v) for k, v in out.items() if k != "examples_wrong"]; print("wrong examples:"); [print("  ", w) for w in wrong]
