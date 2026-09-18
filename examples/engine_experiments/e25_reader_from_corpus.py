#!/usr/bin/env python3
"""E25: can the SYMBOLIC RULE label real sentences from any corpus for free, and does a probe trained on those labels
TRANSFER where one trained on our templates did not?

e22 measured: the representation probe trained on our 320 TEMPLATED sentences with the rule's labels reads QuaRTz test at
0.57 / 0.88 / 0.57 (0.5B / 1.5B / 3B) — it inherits the template genre; trained on QuaRTz's own 283 HUMAN-labelled training
paragraphs it reads 0.80 / 0.91 / 0.90. Hypothesis under test here: what was missing was not human labels and not quantity,
it was the GENRE of the training sentences — real prose. So: take real sentences from two corpora, let the rule
(graph_engine.polarity_rules.asserted_sign, no human labels anywhere) label the ones it can read, and train the same probe
on those.

Training sources (both rule-labelled, classes balanced by subsampling the majority sign, seed 0):
  (A) arXiv hep-ex abstracts (7201 abstracts, $HUNT_DATA/arxiv_hepex.jsonl, the e23 corpus). Sentences are taken from the
      abstract, sentences containing TeX ($, backslash, braces) or shorter than 6 / longer than 45 words are dropped, and
      the quantity pair is found by ONE documented heuristic: find the first DIRECTION VERB (increase/reduce/fall/... —
      the verb forms of the rule's UP/DOWN lexicon, not the comparatives); among the maximal runs of "noun-ish" tokens
      (tokens that are not function words, not direction words, and at least 3 characters) that lie entirely before it and
      within 8 tokens, take the LONGEST (ties: the nearest), and likewise the longest such run after it; each is trimmed to
      at most 4 tokens (the tail of the span before, the head of the span after). x = the span before, y = the span after.
      The pair is then handed with the sentence to the rule; the sentence is kept only when the rule answers (non-zero).
      The pair is a guess made by a heuristic, not an annotation: the hand check below counts how often the triple
      (x, y, sign) is actually right.
  (B) QuaRTz TRAIN paragraphs (the test split is never touched). Sentence-split; the pair is the annotated property pair
      (cause_prop, effect_prop) of the paragraph; the sign is again the RULE's, not QuaRTz's human label.
  (C) A and B together.

Test sets (unchanged from the earlier experiments):
  * QuaRTz TEST split, 81 paragraphs, human labels, the e22 protocol verbatim (one item per para_id, truth =
    direction(cause) x direction(effect)).
  * the e7 320 templated sentences, overall and split into the half that AGREES with textbook physics and the half that
    CONTRADICTS it (the e20 test).
The layer of every probe is chosen on its own training data only (inner split over the (x, y) groups).

Run per model: JUDGE_MODEL=<dir> JUDGE_TAG=_1p5b E7_DEVICE=cuda python3 e25_reader_from_corpus.py
(writes e25_results<TAG>.json, then merges every such file into e25_results.json and prints the table).
"""
import contextlib, glob, importlib, io, json, os, random, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"] + "/datasets")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    e7 = importlib.import_module("e7_typed_judge_measured")
from graph_engine.polarity_rules import asserted_sign, UP, DOWN
from graph_engine.representation_probe import RepresentationProbe
from datasets import load_dataset

HERE = Path(__file__).parent
TAG = e7.TAG or "_0p5b"
NAME = Path(e7.MODEL).name
CORPUS = Path(os.environ.get("HUNT_DATA", str(ROOT / "data"))) / "arxiv_hepex.jsonl"
SEED = 0

# ------------------------------------------------------------------ mining real sentences
WORD = re.compile(r"[a-z0-9']+")
# direction VERBS only: the comparatives in the rule's lexicon ("higher", "more", "larger") are mostly not relational in
# an abstract ("higher energies", "more data"), so they are not used to trigger a candidate pair.
VERBS = {"increase", "increases", "increased", "increasing", "decrease", "decreases", "decreased", "decreasing",
         "rise", "rises", "rising", "rose", "fall", "falls", "falling", "fell", "grow", "grows", "growing", "grew",
         "drop", "drops", "dropping", "dropped", "reduce", "reduces", "reduced", "reducing", "raise", "raises",
         "raised", "raising", "lower", "lowers", "lowered", "lowering", "boost", "boosts", "climb", "climbs",
         "shrink", "shrinks", "decline", "declines"}
STOP = set("""the a an of in on at to for with by from as is are was were be been being this that these those it they he she we our their its
and or but not no never if when while which who whom whose than then so such very most much many few each any all both other some
we investigate present report study measure observe show shows shown find finds found using used use based data results result analysis
have has had do does did can could may might will would should must about into over under between within per also however thus
therefore here there new first second same only because due during after before via""".split())
BAD = STOP | UP | DOWN


def sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())]


def noun_spans(tokens):
    """maximal runs of noun-ish tokens (not function words, not direction words, >= 3 characters) as (start, end)."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] in BAD or len(tokens[i]) < 3:
            i += 1
            continue
        j = i
        while j < len(tokens) and tokens[j] not in BAD and len(tokens[j]) >= 3:
            j += 1
        out.append((i, j))
        i = j
    return out


def candidate_pair(sent: str, window: int = 8, maxlen: int = 4):
    """The heuristic (x, y) of one hep-ex sentence; None when the sentence has no direction verb with spans on both sides."""
    toks = WORD.findall(sent.lower())
    dirs = [k for k, t in enumerate(toks) if t in VERBS]
    if not dirs:
        return None
    d = dirs[0]
    sp = noun_spans(toks)
    before = [s for s in sp if s[1] <= d and d - s[1] <= window]
    after = [s for s in sp if s[0] > d and s[0] - d <= window]
    if not before or not after:
        return None
    pick = lambda ss: max(ss, key=lambda s: (min(s[1] - s[0], maxlen), -abs(s[0] - d)))
    b, a = pick(before), pick(after)
    x = " ".join(toks[max(b[0], b[1] - maxlen):b[1]])
    y = " ".join(toks[a[0]:min(a[1], a[0] + maxlen)])
    return (x, y) if x and y and x != y else None


def usable(sent: str) -> bool:
    return not any(c in sent for c in "$\\{}") and 6 <= len(sent.split()) <= 45


def mine_hepex(path):
    """Every kept sentence of the corpus with its heuristic pair; `sign` 0 = the rule abstains."""
    rows = []
    for line in open(path):
        rec = json.loads(line)
        for sent in sentences(rec["abstract"]):
            if not usable(sent):
                continue
            pair = candidate_pair(sent)
            if pair is None:
                rows.append({"id": rec["id"], "text": sent, "x": None, "y": None, "sign": 0})
                continue
            rows.append({"id": rec["id"], "text": sent, "x": pair[0], "y": pair[1],
                         "sign": int(asserted_sign(sent, pair[0], pair[1]))})
    return rows


def quartz_rows():
    """e22/e15's construction verbatim: one item per para_id, truth = product of the two annotated MORE/LESS signs."""
    ds = load_dataset("allenai/quartz")
    rows, seen = [], set()
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            a = ex["para_anno"]
            a = eval(a) if isinstance(a, str) else a
            cp, ep = a.get("cause_prop", "").strip(), a.get("effect_prop", "").strip()
            cs, es = a.get("cause_dir_sign", ""), a.get("effect_dir_sign", "")
            if not (cp and ep and cs in ("MORE", "LESS") and es in ("MORE", "LESS")) or ex["para_id"] in seen:
                continue
            seen.add(ex["para_id"])
            rows.append((ex["para"], cp, ep, (1 if cs == "MORE" else -1) * (1 if es == "MORE" else -1), split))
    return rows


def mine_quartz_train(rows):
    """Sentences of the TRAIN paragraphs with the annotated property pair, labelled by the RULE (human label kept only to
    report how often the rule's label agrees with it — it is never used for training)."""
    out = []
    for para, cp, ep, human, split in rows:
        if split != "train":
            continue
        for sent in sentences(para):
            s = int(asserted_sign(sent, cp, ep))
            if s:
                out.append({"text": sent, "x": cp, "y": ep, "sign": s, "human_para_label": int(human)})
    return out


def balance(rows, seed=SEED):
    pos = [r for r in rows if r["sign"] > 0]
    neg = [r for r in rows if r["sign"] < 0]
    n = min(len(pos), len(neg))
    rng = random.Random(seed)
    return rng.sample(pos, n) + rng.sample(neg, n)


# ------------------------------------------------------------------ data
hepex_all = mine_hepex(CORPUS)
hepex_lab = [r for r in hepex_all if r["sign"]]
qrows = quartz_rows()
quartz_lab = mine_quartz_train(qrows)
train_A, train_B = balance(hepex_lab), balance(quartz_lab)
train_C = train_A + train_B

qtexts = [r[0] for r in qrows]; qpairs = [(r[1], r[2]) for r in qrows]
qtruth = np.array([r[3] for r in qrows]) > 0; qsplit = np.array([r[4] for r in qrows])
te = qsplit == "test"
sub = lambda a, m: [v for v, k in zip(a, m) if k]
q_texts, q_pairs, q_truth = sub(qtexts, te), sub(qpairs, te), qtruth[te]

items = e7.items                                            # the 320 templated sentences
t_texts = [t for t, x, y, *_ in items]; t_pairs = [(x, y) for t, x, y, *_ in items]
t_truth = np.array([s > 0 for _, _, _, s, _, _ in items]); t_agree = np.array([a for *_, a, _ in items])

# ------------------------------------------------------------------ probes (hidden states cached across the three fits)
CACHE: dict = {}


class CachedProbe(RepresentationProbe):
    """Same class; the last-prompt-token hidden states of a (text, x, y) are computed once per model run."""

    def hidden_states(self, texts, pairs, layer=None):
        keys = [(t, x, y) for t, (x, y) in zip(texts, pairs)]
        miss = [k for k in dict.fromkeys(keys) if k not in CACHE]
        if miss:
            H = RepresentationProbe.hidden_states(self, [k[0] for k in miss], [(k[1], k[2]) for k in miss])
            for k, h in zip(miss, H):
                CACHE[k] = h
        out = np.stack([CACHE[k] for k in keys])
        return out if layer is None else out[:, layer]


def fit(rows):
    p = CachedProbe((e7.tok, e7.model), device=e7.DEV, batch_size=8, name=NAME)
    return p.fit([r["text"] for r in rows], [(r["x"], r["y"]) for r in rows],
                 [r["sign"] > 0 for r in rows], seed=SEED)


def evaluate(p):
    pq = p.predict_proba(q_texts, q_pairs) > 0.5
    pt = p.predict_proba(t_texts, t_pairs) > 0.5
    return {"layer": int(p.layer),
            "quartz_test": round(float((pq == q_truth).mean()), 3),
            "templates_all": round(float((pt == t_truth).mean()), 3),
            "templates_agree_physics": round(float((pt == t_truth)[t_agree].mean()), 3),
            "templates_contradict_physics": round(float((pt == t_truth)[~t_agree].mean()), 3)}


probes = {"a_hepex_rule_labelled": train_A, "b_quartz_train_rule_labelled": train_B, "c_both": train_C}
read = {}
for key, rows in probes.items():
    p = fit(rows)
    read[key] = dict(n_train=len(rows), **evaluate(p))
    print(key, read[key], flush=True)

# the rule itself on the two test sets (for the same table)
rule_q = np.array([asserted_sign(t, x, y) for t, (x, y) in zip(q_texts, q_pairs)])
ans = rule_q != 0

# ------------------------------------------------------------------ hand check: 20 random rule-labelled hep-ex sentences
rng = random.Random(25)
sample = rng.sample(hepex_lab, 20)
# Verdicts after reading all 20 (index in the sample -> is the TRIPLE (x, y, sign) a correct reading of the sentence?).
# "True" = both spans are quantities of the sentence AND the sentence asserts that sign between them; "False" = it does
# not — almost always because the heuristic's x span is not a quantity at all ("findings reveal", "work highlights",
# "establishing") or because the two spans are objects of different verbs and no relation between them is asserted.
# Items 8 and 15 are generous calls (the spans carry stray tokens but name the right things); 7 has x and y swapped
# relative to the sentence's cause/effect, which does not change the sign of a monotone relation.
HAND = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: True, 8: True, 9: False,
        10: False, 11: False, 12: False, 13: False, 14: False, 15: True, 16: False, 17: False, 18: True, 19: False}
# A weaker criterion, also counted by hand on the same 20: does the SIGN match the direction the sentence gives to the
# y span (ignoring whether x is a real quantity)? That is what the probe actually sees a label for.
HAND_Y_DIRECTION = {0: True, 1: True, 2: False, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True,
                    10: False, 11: True, 12: True, 13: False, 14: True, 15: True, 16: True, 17: False, 18: True, 19: False}

res = {
    "model": NAME, "size_tag": TAG.lstrip("_"), "device": e7.DEV,
    "mining": {
        "hepex_sentences_kept": len(hepex_all),
        "hepex_with_candidate_pair": sum(1 for r in hepex_all if r["x"]),
        "hepex_rule_answers": len(hepex_lab),
        "hepex_rule_coverage_of_kept_sentences": round(len(hepex_lab) / len(hepex_all), 4),
        "hepex_rule_coverage_where_a_pair_was_found": round(len(hepex_lab) / max(1, sum(1 for r in hepex_all if r["x"])), 4),
        "hepex_sign_balance": {"plus": sum(1 for r in hepex_lab if r["sign"] > 0), "minus": sum(1 for r in hepex_lab if r["sign"] < 0)},
        "quartz_train_paragraphs": sum(1 for r in qrows if r[4] == "train"),
        "quartz_train_rule_answers": len(quartz_lab),
        "quartz_train_sign_balance": {"plus": sum(1 for r in quartz_lab if r["sign"] > 0), "minus": sum(1 for r in quartz_lab if r["sign"] < 0)},
        "quartz_train_rule_label_agrees_with_human_para_label": round(
            float(np.mean([(r["sign"] > 0) == (r["human_para_label"] > 0) for r in quartz_lab])), 3),
        "balanced_train_sizes": {"a_hepex": len(train_A), "b_quartz_train": len(train_B), "c_both": len(train_C)},
    },
    "probes": read,
    "rule_on_quartz_test": {"coverage": round(float(ans.mean()), 3),
                            "accuracy_where_answered": round(float(((rule_q > 0) == q_truth)[ans].mean()), 3)},
    "test_sets": {"quartz_test_n": int(len(q_truth)), "quartz_test_majority": round(float(max(q_truth.mean(), 1 - q_truth.mean())), 3),
                  "templates_n": len(items)},
    "e22_reference": {"a_our_templates_rule_labels": [0.57, 0.88, 0.57], "b_quartz_train_human_labels": [0.80, 0.91, 0.90],
                      "c_token_output": [0.83, 0.89, 0.93], "order": ["0.5B", "1.5B", "3B"]},
    "hand_check_20_hepex": {"seed": 25,
                            "items": [{"i": i, "id": r["id"], "x": r["x"], "y": r["y"], "sign": r["sign"], "text": r["text"],
                                       "triple_right": HAND.get(i), "y_direction_right": HAND_Y_DIRECTION.get(i)}
                                      for i, r in enumerate(sample)],
                            "n_triples_right": sum(1 for v in HAND.values() if v),
                            "n_y_direction_right": sum(1 for v in HAND_Y_DIRECTION.values() if v),
                            "criterion": "triple_right = both spans are quantities and the sentence asserts that sign "
                                         "between them; y_direction_right = the sign matches the direction the sentence "
                                         "gives the y span, whatever x is"},
    "note": "all training labels come from graph_engine.polarity_rules.asserted_sign; no human label is used for fitting. "
            "QuaRTz test is touched only at prediction; the layer of each probe is chosen on its own training data.",
}
json.dump(res, open(HERE / f"e25_results{TAG}.json", "w"), indent=1)

order = {"_0p5b": 0, "_1p5b": 1, "_3b": 2}
files = sorted(glob.glob(str(HERE / "e25_results_*.json")), key=lambda f: order.get("_" + Path(f).stem.split("results_")[1], 9))
merged = {"experiment": "e25", "question": "does a probe trained on RULE-LABELLED REAL sentences transfer where one trained on templates did not?",
          "by_model": [json.load(open(f)) for f in files]}
json.dump(merged, open(HERE / "e25_results.json", "w"), indent=1)
hdr = f"{'model':<8}{'train source':<30}{'QuaRTz test':>12}{'tmpl all':>10}{'agree':>8}{'contra':>8}{'n_train':>9}{'layer':>7}"
print(hdr); print("-" * len(hdr))
for r in merged["by_model"]:
    for k, v in r["probes"].items():
        print(f"{r['size_tag']:<8}{k:<30}{v['quartz_test']:>12.3f}{v['templates_all']:>10.3f}"
              f"{v['templates_agree_physics']:>8.3f}{v['templates_contradict_physics']:>8.3f}{v['n_train']:>9}{v['layer']:>7}")
for k, v in res["mining"].items():
    print(k, v)
print("rule on quartz test:", res["rule_on_quartz_test"])
