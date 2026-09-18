#!/usr/bin/env python3
"""E23: the symbolic numeric channel (graph_engine.numeric_rules) on a REAL corpus of arXiv hep-ex abstracts, fed into
margin_net's lineage-aware GLS and its χ² "do these reports estimate one number?" test.

Corpus. The arXiv API (https://export.arxiv.org/api/query, terms allow programmatic access with a polite delay; 3 s
between requests here). Two parts, both cached in $HUNT_DATA/arxiv_hepex.jsonl (outside the repo):
  (a) the 6000 most recent cat:hep-ex abstracts — the sample coverage is measured on;
  (b) targeted all-years queries (abs:"top quark mass", abs:"W boson mass", abs:"Higgs boson mass") so that the
      per-quantity groups span Tevatron-era and LHC-era papers, which (a) alone does not.

Pipeline. abstract → extract_numeric_claims → keep the records whose quantity phrase matches one of the patterns below
AND whose value is inside the stated window in the stated unit (the window is a coarse filter against homonyms, e.g.
"m_t" used for a transverse mass; it is a selection and is reported as such) → one report per paper (the one with the
smallest σ) → margin_net edge, twice:
    per_paper          every paper is its own root source  (N_eff = n)
    shared_systematics papers of one collaboration share rho = 0.5 of their error variance (margin_net's third lineage
                       state, added after the first run showed the copy reading is wrong for successive measurements)
    per_collaboration  papers of one collaboration (ATLAS/CMS/CDF/D0/LHCb/ALICE/BaBar/Belle/LEP experiments, detected
                       from the text) derive from one root; papers with no detected collaboration stay their own root.
Reported per quantity: n, m̂ ± s, N_eff in both modes, Q/dof and its p-value, and the PDG value fetched from pdglive.

Honest caveats: abstracts mix pole/MC masses, partial and combined results, and re-quotes of other experiments'
numbers; the χ² is therefore expected to be small — the interesting question is whether it sees the known
Tevatron/LHC offset, which is checked separately (`top_mass_by_era`).
"""
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.margin_net import MarginNet                      # noqa: E402
from graph_engine.numeric_rules import extract_numeric_claims      # noqa: E402

from _data import DATA
CACHE = DATA / "arxiv_hepex.jsonl"
OUT = Path(__file__).with_name("e23_results.json")
API = ("https://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate&sortOrder={o}"
       "&start={s}&max_results={n}")
QUERIES = [("cat:hep-ex", "descending", 6000),
           ('cat:hep-ex+AND+abs:%22top+quark+mass%22', "ascending", 1500),
           ('cat:hep-ex+AND+abs:%22W+boson+mass%22', "ascending", 800),
           ('cat:hep-ex+AND+abs:%22Higgs+boson+mass%22', "ascending", 800),
           ('cat:hep-ex+AND+abs:%22top+quark+mass%22', "descending", 1000),
           ('cat:hep-ex+AND+abs:%22W+boson+mass%22', "descending", 500),
           ('cat:hep-ex+AND+abs:%22Higgs+boson+mass%22', "descending", 800),
           ('cat:hep-ex+AND+abs:%22mass+of+the+Higgs+boson%22', "descending", 800)]
PAGE, DELAY = 500, 3.0
_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)

# quantity phrase pattern, unit, [value window], name
TARGETS = {
    "top_quark_mass": (re.compile(r"(?:^|[^a-z])m[_\s]?(?:t|top)\b|top[-\s]?quark mass|top mass|mass of the top", re.I),
                       "GeV", (150.0, 200.0), 10.0),
    "W_boson_mass": (re.compile(r"(?:^|[^a-z])m[_\s]?w\b|w[-\s]?boson mass|mass of the w", re.I),
                     "GeV", (78.0, 83.0), 1.0),
    "Higgs_boson_mass": (re.compile(r"(?:^|[^a-z])m[_\s]?h\b|higgs[\w\s-]{0,20}mass|mass of the higgs", re.I),
                         "GeV", (120.0, 130.0), 5.0),
}
PDG = {  # fetched from pdglive.lbl.gov (see pdg_lookup), values pinned here for reproducibility
    "top_quark_mass": {"value": 172.60, "sigma": 0.27, "node": "Q007TP",
                       "note": "OUR AVERAGE, LHC+Tevatron; TEVEWWG Tevatron average quoted there is 174.30 +- 0.35 +- 0.54"},
    "W_boson_mass": {"value": 80.3625, "sigma": 0.0077, "node": "S043M",
                     "note": "OUR EVALUATION, LHC-TeVatron M_W Working Group"},
    "Higgs_boson_mass": {"value": 125.13, "sigma": 0.11, "node": "S126M",
                         "note": "OUR AVERAGE, error includes scale factor 1.5"},
}
COLLABS = ["ATLAS", "CMS", "LHCb", "ALICE", "CDF", "D0", "DZero", "BaBar", "Belle", "OPAL", "DELPHI", "ALEPH", "L3",
           "H1", "ZEUS", "NA62", "T2K", "NOvA", "MINOS", "STAR", "PHENIX"]
_COLLAB_RE = {c: re.compile(rf"\b{c}\b", re.I if c in ("DZero",) else 0) for c in COLLABS}


# -- corpus --------------------------------------------------------------------------------------
def _field(chunk, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", chunk, re.S)
    if not m:
        return ""
    t = re.sub(r"<.*?>", " ", m.group(1))
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def fetch_corpus():
    """Cache-first; only the missing part of each query is requested, 3 s apart."""
    recs, seen = [], set()
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            q = r.get("query", "cat:hep-ex")
            if "|" not in q:                               # legacy cache written before the sort order was keyed
                r["query"] = q + ("|descending" if q == "cat:hep-ex" else "|ascending")
            if r["id"] not in seen:
                seen.add(r["id"]); recs.append(r)
    for q, order, target in QUERIES:
        key = f"{q}|{order}"
        have = sum(1 for r in recs if r["query"] == key)
        start = have
        while have < target:
            url = API.format(q=q, o=order, s=start, n=PAGE)
            try:
                raw = urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "replace")
            except Exception as e:                         # noqa: BLE001
                print("  fetch failed:", e, flush=True); break
            chunks = _ENTRY.findall(raw)
            if not chunks:
                break
            for c in chunks:
                rid = _field(c, "id")
                if rid in seen:
                    continue
                seen.add(rid)
                recs.append({"id": rid, "published": _field(c, "published"), "title": _field(c, "title"),
                             "abstract": _field(c, "summary"), "query": key,
                             "authors": re.findall(r"<name>(.*?)</name>", c)[:6]})
            start += len(chunks); have += len(chunks)
            print(f"  {q} start={start} corpus={len(recs)}", flush=True)
            with CACHE.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            time.sleep(DELAY)
    return recs


def pdg_lookup(node):
    """The PDG value as pdglive shows it; returns the raw text around OUR AVERAGE/FIT/EVALUATION for checking."""
    url = f"https://pdglive.lbl.gov/DataBlock.action?node={node}"
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")))
    m = re.search(r"OUR (?:AVERAGE|FIT|EVALUATION)", t)
    return t[max(0, m.start() - 160):m.start() + 60] if m else ""


# -- extraction ----------------------------------------------------------------------------------
def collaboration(rec):
    txt = rec["title"] + " " + rec["abstract"] + " " + " ".join(rec.get("authors", []))
    hits = [c for c, r in _COLLAB_RE.items() if r.search(txt)]
    if "DZero" in hits and "D0" not in hits:
        hits = ["D0"] + [h for h in hits if h != "DZero"]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def year(rec):
    return int(rec["published"][:4]) if rec.get("published") else 0


def collect(recs):
    per_abstract, groups = [], {k: {} for k in TARGETS}
    for rec in recs:
        cl = extract_numeric_claims(rec["abstract"])
        per_abstract.append(len(cl))
        for c in cl:
            for name, (pat, unit, (lo, hi), smax) in TARGETS.items():
                if c["unit"] == unit and lo <= c["value"] <= hi and 0 < c["sigma"] < smax \
                        and "range_uniform" not in c["flags"] and pat.search(c["quantity"]):
                    g = groups[name]
                    if rec["id"] not in g or c["sigma"] < g[rec["id"]]["claim"]["sigma"]:
                        g[rec["id"]] = {"claim": c, "rec": rec}
    return per_abstract, groups


RHO_SHARED = 0.5


def run_net(items, mode):
    net = MarginNet()
    reports = []
    for pid, it in items:
        collab = it["collab"]
        if mode == "per_collaboration" and collab:
            net.add_sources([{"id": pid, "derives_from": [collab]}])
        elif mode == "shared_systematics" and collab:                 # third lineage state: rho of the variance shared within a collaboration
            net.add_sources([{"id": pid, "shares": {collab: RHO_SHARED}}])
        reports.append({"margin": it["claim"]["value"], "sigma": it["claim"]["sigma"], "sources": [pid]})
    net.add_edge("q", ["q"], reports)
    e = net.estimate("q")
    return {"m_hat": round(e.m, 4), "s": round(e.s, 4), "n_eff": round(e.n_eff, 2),
            "chi2_Q": round(e.q, 2), "dof": e.dof, "p_agree": float(f"{e.p_agree:.3g}"), "kind": e.kind}


# -- the hand check (30 random abstracts, seed 24, read one by one by the author) -------------------
# Every abstract in the sample was read; the 24 not listed here contain no number with a stated uncertainty and the
# extractor correctly produced nothing. Verdicts are per abstract, keyed by arXiv id so they cannot drift silently.
HAND_VERDICTS = {
 "http://arxiv.org/abs/2502.03277v2": ("right", "(0.49^{+1.56}_{-0.38}) nb -> 0.49 +- 1.56 nb, quantity 'total cross sections', asymmetric flagged"),
 "http://arxiv.org/abs/2602.01759v1": ("right", "'mass ranges 1.6-3.1 GeV' -> midpoint 2.35 +- 0.43 GeV, range_uniform flagged"),
 "http://arxiv.org/abs/2609.00152v1": ("wrong", "numbers right (0.2-15 GeV range) but the quantity phrase is junk: 'axion-like particles covering the'"),
 "http://arxiv.org/abs/1109.1490v1": ("abstained_should_have", "custom macro $\\mtop = \\gevcc{\\measStatSyst{172.3}{2.4}{1.0}}$ — paper-private macros are not covered"),
 "http://arxiv.org/abs/2006.13251v2": ("abstained_should_have", "'$\\mu_\\mathrm{H}$ $=$ 3.7 $\\pm$ 1.2 (stat)...': the quantity is a single Greek letter with a subscript, which the phrase rule rejects"),
 "http://arxiv.org/abs/2507.12598v2": ("abstained_should_have", "'$\\mu=0.9^{+0.7}_{-0.6}$': same single-Greek-letter quantity"),
}


def hand_sample(recs, n=30, seed=24):
    rng = random.Random(seed)
    picked = rng.sample(recs, n)
    out = []
    for r in picked:
        cl = extract_numeric_claims(r["abstract"])
        out.append({"id": r["id"], "abstract": r["abstract"],
                    "extractions": [{k: c[k] for k in ("value", "sigma", "unit", "quantity", "flags")} for c in cl],
                    "spans": [r["abstract"][a:b] for a, b in (c["span"] for c in cl)]})
    return out


def main():
    print("corpus...", flush=True)
    recs = fetch_corpus()
    base = [r for r in recs if r["query"].startswith("cat:hep-ex|")]
    print(f"corpus {len(recs)} abstracts ({len(base)} in the plain cat:hep-ex sample)", flush=True)

    per_abstract, groups = collect(recs)
    base_counts = [len(extract_numeric_claims(r["abstract"])) for r in base]
    res = {"corpus": {"n_abstracts": len(recs), "n_recent_hepex": len(base),
                      "years": [min(year(r) for r in recs), max(year(r) for r in recs)],
                      "source": "arXiv API export.arxiv.org, queries " + "; ".join(sorted({q for q, _, _ in QUERIES}))},
           "coverage": {"fraction_with_at_least_one_claim_recent_sample": round(sum(c > 0 for c in base_counts) / max(len(base), 1), 3),
                        "fraction_whole_corpus": round(sum(c > 0 for c in per_abstract) / max(len(per_abstract), 1), 3),
                        "claims_total": sum(per_abstract), "claims_per_abstract_mean": round(sum(per_abstract) / max(len(per_abstract), 1), 2)},
           "quantities": {}}

    for name, g in groups.items():
        items = []
        for pid, it in g.items():
            it["collab"] = collaboration(it["rec"])
            items.append((pid, it))
        if len(items) < 2:
            res["quantities"][name] = {"n_reports": len(items)}
            continue
        collabs = {}
        for _, it in items:
            collabs[it["collab"] or "(none)"] = collabs.get(it["collab"] or "(none)", 0) + 1
        res["quantities"][name] = {
            "n_reports": len(items),
            "collaborations": dict(sorted(collabs.items(), key=lambda kv: -kv[1])),
            "per_paper": run_net(items, "per_paper"),
            "per_collaboration": run_net(items, "per_collaboration"),
            "shared_systematics_rho0.5": run_net(items, "shared_systematics"),
            "pdg": PDG[name],
            "values_min_max": [round(min(it["claim"]["value"] for _, it in items), 3),
                               round(max(it["claim"]["value"] for _, it in items), 3)],
        }

    # the era question: do the top-mass reports across years estimate ONE number?
    g = groups["top_quark_mass"]
    eras = {"upto_2011_tevatron_era": [], "from_2012_lhc_era": []}
    for pid, it in g.items():
        it["collab"] = it.get("collab") or collaboration(it["rec"])
        (eras["upto_2011_tevatron_era"] if year(it["rec"]) <= 2011 else eras["from_2012_lhc_era"]).append((pid, it))
    res["top_mass_by_era"] = {k: (run_net(v, "per_paper") | {"n": len(v)}) if len(v) >= 2 else {"n": len(v)}
                              for k, v in eras.items()}
    tev = [it["claim"]["value"] for _, it in eras["upto_2011_tevatron_era"]]
    lhc = [it["claim"]["value"] for _, it in eras["from_2012_lhc_era"]]
    res["top_mass_by_era"]["median_tevatron_era"] = round(sorted(tev)[len(tev) // 2], 3) if tev else None
    res["top_mass_by_era"]["median_lhc_era"] = round(sorted(lhc)[len(lhc) // 2], 3) if lhc else None

    items = hand_sample(recs)
    counts = {"right": 0, "wrong": 0, "abstained_should_have": 0, "correct_abstention": 0, "unjudged": 0}
    for it in items:
        v = HAND_VERDICTS.get(it["id"])
        if v:
            counts[v[0]] += 1
        elif it["extractions"]:
            counts["unjudged"] += 1                        # an extraction the author did not judge: must stay 0
        else:
            counts["correct_abstention"] += 1
    res["hand_check"] = {"n_abstracts": 30, "seed": 24, "counts": counts,
                         "n_extractions_in_sample": sum(len(it["extractions"]) for it in items),
                         "verdicts": {k: list(v) for k, v in HAND_VERDICTS.items()},
                         "note": "every sampled abstract was read; 'correct_abstention' = no number with a stated "
                                 "uncertainty in the text and the extractor produced nothing"}
    res["hand_check_items"] = items
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("hand_check_items",)}, indent=1))


if __name__ == "__main__":
    main()
