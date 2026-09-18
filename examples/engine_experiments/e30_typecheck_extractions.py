#!/usr/bin/env python3
"""E30: the checker (graph_engine.claim_types.check_numeric) run over EVERY numeric record the symbolic extractor
produces on the two real corpora — arXiv hep-ex abstracts and the PubMed blood-pressure abstracts — to measure how much
of what a proposer proposes is not even well-typed, and to say why.

Corpora (cached in $HUNT_DATA, outside the repo, by e23 and e26): arxiv_hepex.jsonl (arXiv API, cat:hep-ex plus
targeted mass queries) and pubmed_*.jsonl (PubMed E-utilities, blood-pressure trials and their re-analyses).
Pipeline: title + abstract → numeric_rules.extract_numeric_claims → claim_types.check_numeric. Nothing is filtered,
selected or grouped first: this is the raw proposer output, which is the population a graph would ingest.

Reported: share rejected per corpus, the reason histogram, and the reason histogram crossed with the extractor's own
flags. Then a HAND CHECK: 20 rejected and 20 accepted records drawn with a fixed seed, read by the author against the
sentence they came from, each marked agree / disagree with the checker. The verdicts are in HAND below (keyed by
corpus and span so a rerun reproduces them) and the agreement rates are computed from them, not asserted.

Honest reading of the numbers: `unchecked_quantity` is the modal ACCEPT — the canonical table is small, so most
accepts mean "no type claim was made", not "verified". The rejects are the load-bearing half, and their hand check is
what says whether the rule is right.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_engine.claim_types import check_numeric                 # noqa: E402
from graph_engine.numeric_rules import extract_numeric_claims      # noqa: E402

from _data import DATA                                             # noqa: E402

OUT = Path(__file__).with_name("e30_results.json")
CORPORA = {"hep-ex": ["arxiv_hepex.jsonl"],
           "pubmed": sorted(p.name for p in DATA.glob("pubmed_*.jsonl"))}
SEED, N_SAMPLE, N_HAND = 30, 20, 10        # 20 drawn per class per corpus, the first 10 of each read by hand

# the author's verdict on each sampled record, filled after reading the sentence it came from:
# key = "<corpus>|<doc index>|<span start>", value = "agree" | "disagree" + a note
HAND: dict[str, str] = {
    # -- hep-ex, REJECTED (read against the sentence each came from)
    "hep-ex|R0": "agree: Delta m^2_21 = 7.96e-5 eV^2 — the extractor lost 'eV$^2$', so the record really has no unit; the label 'mass' is imprecise (it is a mass squared), the verdict is not",
    "hep-ex|R1": "agree: m_t = 174.95 GeV with the unit macro lost by the extractor; a mass with no unit cannot enter",
    "hep-ex|R2": "agree: Gy (gray) is outside the grammar — abstaining is the right move, and it is a grammar gap, not a defect of the record",
    "hep-ex|R3": "agree: integrated luminosity 186 nb^{-1}, the extractor dropped the ^{-1}; L^2 against L^-2 catches exactly that",
    "hep-ex|R4": "agree: 'LHC' read as a unit out of 'Run 1-2 LHC data taking' — a spurious record, rightly blocked",
    "hep-ex|R5": "DISAGREE: the quantity is a scattering length in fm, 'extrapolated to the physical pion mass' only ends in the word mass; surface-phrase matching fails here",
    "hep-ex|R6": "agree: same dropped ^{-1} on fb, integrated luminosity",
    "hep-ex|R7": "agree: 'top quark mass ... GeV' with the unit not captured after the asymmetric term",
    "hep-ex|R8": "agree: Bq/m^3 truncated to 'Bq/m' by the extractor; the grammar does not know Bq either",
    "hep-ex|R9": "DISAGREE: a cross section quoted RELATIVE to the standard model is dimensionless; the checker demanded an area unit",
    # -- hep-ex, ACCEPTED
    "hep-ex|A0": "agree: 'VUV' resolves to nothing, so no type claim is made — accepted as unchecked, not as verified",
    "hep-ex|A1": "agree: unchecked; the quantity phrase is prose debris, which is an extractor defect the checker says nothing about",
    "hep-ex|A2": "agree: masses in MeV, dimension matches M under natural units",
    "hep-ex|A3": "agree: an energy range in MeV",
    "hep-ex|A4": "agree: mtop = 172.6 GeV",
    "hep-ex|A5": "agree: '\left' is debris, unchecked",
    "hep-ex|A6": "agree: a ratio of branching fractions, no unit, unchecked",
    "hep-ex|A7": "agree: m_top in GeV",
    "hep-ex|A8": "agree: '\pm' is debris, unchecked",
    "hep-ex|A9": "agree on the type (an efficiency is dimensionless) — but note the checker cannot see that the % was lost, so 37.4 stands where 0.374 belongs: a SCALE error no dimension check catches",
    # -- pubmed, REJECTED
    "pubmed|R0": "agree: DBP with 'mm Hg' written with a space, so the record carries no unit at all",
    "pubmed|R1": "agree: 'mm Hg' split, the record claims metres for a blood pressure — the mismatch is real",
    "pubmed|R2": "agree: age in years, but numeric_rules refuses lowercase words of 4+ letters as units, so the record has none",
    "pubmed|R3": "agree: 'vs' read as a unit",
    "pubmed|R4": "agree: same missing 'years' on an age",
    "pubmed|R5": "agree: same missing 'years' on an age",
    "pubmed|R6": "agree: 'vs' read as a unit",
    "pubmed|R7": "agree: mean follow-up with 'years' dropped",
    "pubmed|R8": "agree: mean age with 'years' dropped",
    "pubmed|R9": "agree: 'but' read as a unit on a mean difference",
    # -- pubmed, ACCEPTED
    "pubmed|A0": "agree: HR, no unit",
    "pubmed|A1": "agree: Cohen's d is not in the table, unchecked",
    "pubmed|A2": "agree: HR, no unit",
    "pubmed|A3": "agree: relative risk, no unit",
    "pubmed|A4": "agree: OR, no unit",
    "pubmed|A5": "agree: 'per 1 SD' makes the phrase a product, on which the table abstains",
    "pubmed|A6": "DISAGREE: 6.2 mm Hg with 'mm Hg' split, so the unit is metres — but the quantity phrase is a drug comparison that resolves to nothing, so the checker accepts it as unchecked. The same defect R1 catches goes through here",
    "pubmed|A7": "agree: 'cardiovascular' resolves to nothing, unchecked",
    "pubmed|A8": "agree: odds ratio, no unit",
    "pubmed|A9": "agree: RR, no unit",
}


def docs(files):
    for name in files:
        f = DATA / name
        if not f.exists():
            continue
        for i, line in enumerate(f.open()):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = " ".join(x for x in (d.get("title"), d.get("abstract")) if x)
            if text:
                yield f"{name}:{i}", text


def run():
    out = {"seed": SEED, "corpora": {}}
    for corpus, files in CORPORA.items():
        n_docs = 0
        records = []
        for doc_id, text in docs(files):
            n_docs += 1
            for r in extract_numeric_claims(text):
                c = check_numeric(r)
                a, b = r["span"]
                records.append({"doc": doc_id, "span": [a, b], "value": r["value"], "sigma": r["sigma"],
                                "unit": r["unit"], "quantity": r["quantity"], "log_scale": r["log_scale"],
                                "flags": r["flags"], "ok": c["ok"], "reason": c["reason"],
                                "quantity_kind": c["quantity_kind"], "dimension": c["dimension"],
                                "context": text[max(0, a - 120):b + 40].replace("\n", " ")})
        rejected = [r for r in records if not r["ok"]]
        accepted = [r for r in records if r["ok"]]
        reasons = Counter(r["reason"].split(":")[0] for r in rejected)
        acc_reasons = Counter(r["reason"] for r in accepted)
        by_flag = Counter((r["reason"].split(":")[0], f) for r in rejected for f in (r["flags"] or ["-"]))
        rng = random.Random(SEED)
        samp_r = rng.sample(rejected, min(N_SAMPLE, len(rejected)))[:N_HAND]
        samp_a = rng.sample(accepted, min(N_SAMPLE, len(accepted)))[:N_HAND]
        def tag(kind, i, r):
            key = f"{corpus}|{kind}{i}"
            return {**r, "key": key, "hand": HAND.get(key)}
        out["corpora"][corpus] = {
            "files": files, "n_docs": n_docs, "n_records": len(records), "n_rejected": len(rejected),
            "share_rejected": round(len(rejected) / len(records), 4) if records else None,
            "reject_reasons": dict(reasons.most_common()),
            "accept_reasons": dict(acc_reasons.most_common()),
            "reject_reason_by_extractor_flag": {f"{a}+{b}": n for (a, b), n in by_flag.most_common(20)},
            "hand_check_rejected": [tag("R", i, r) for i, r in enumerate(samp_r)],
            "hand_check_accepted": [tag("A", i, r) for i, r in enumerate(samp_a)],
        }
    # agreement of the hand check with the checker
    agree = Counter()
    for corpus, c in out["corpora"].items():
        for kind in ("hand_check_rejected", "hand_check_accepted"):
            for r in c[kind]:
                h = r["hand"] or ""
                v = "disagree" if h.startswith("DISAGREE") else ("agree" if h.startswith("agree") else "unread")
                agree[(corpus, kind.split("_")[-1], v)] += 1
    out["hand_check_agreement"] = {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(agree.items())}
    OUT.write_text(json.dumps(out, indent=1))
    for corpus, c in out["corpora"].items():
        print(f"{corpus:8s} docs={c['n_docs']:5d} records={c['n_records']:6d} rejected={c['n_rejected']:6d} "
              f"({c['share_rejected']:.1%})  {dict(list(c['reject_reasons'].items())[:6])}")
    print(out["hand_check_agreement"])


if __name__ == "__main__":
    run()
