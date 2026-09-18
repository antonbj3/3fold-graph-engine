#!/usr/bin/env python3
"""E26: the symbolic PROVENANCE channel (graph_engine.provenance_rules) on a REAL medical corpus, fed into
margin_net's lineage-aware GLS together with the numeric channel (graph_engine.numeric_rules).

Why. e23 fed margin_net numbers from hep-ex abstracts and got the lineage from the collaboration name in the text.
In medicine there is no collaboration name: guidelines, reviews and secondary analyses are COPIES of a few trials, so
"many independent reports of the same hazard ratio" is usually one trial counted many times — and the lineage is in
prose ("a secondary analysis of the SPRINT trial", "according to the 2018 ESC/ESH guidelines") or nowhere.
provenance_rules reads that prose; this experiment measures how far it gets on a corpus nobody wrote for it.

Corpus. PubMed via NCBI E-utilities (esearch + efetch, XML), ≤ 3 requests/s, `tool` and `email` parameters as NCBI
asks (email from $NCBI_EMAIL; NCBI asks for one, the repo does not carry one). Seven queries on blood-pressure targets
and intensive control — many numeric results, heavy guideline copying, and (bp_secondary, sprint) the part of the
literature that re-analyses one trial — cached in $HUNT_DATA/pubmed_<topic>.jsonl, outside the repo.

Pipeline, per abstract:
    provenance: extract_provenance  → {kind, target, confidence};  provenance_abstentions → markers with no name
    numbers:    extract_numeric_claims → the records whose quantity phrase names ONE quantity (a hazard ratio for
                all-cause mortality, or one for cardiovascular events) and whose value is a plausible ratio
Then margin_net on that one quantity three ways, changing ONLY the declared lineage:
    independent  every abstract is its own root                       (what a reader who ignores provenance gets)
    copies       abstracts declaring copy/derived/shared from a named trial derive_from that trial
    shared_0.5   the same declarations as `shares: {trial: 0.5}`      (third lineage state, from e23)

Honest caveats. Hazard ratios are combined on the arithmetic scale (margin_net is a linear GLS) with σ from the stated
CI half-width; the log scale would be the right one and the difference is second order for HR near 1, which is where
this corpus sits. Abstracts report different endpoints, populations and follow-ups under the same phrase — the χ² is
expected to fire, and what the three lineage readings change is N_eff and s, not the mixing. The trial named in an
abstract's provenance statement is not necessarily the source of the number quoted in it (module docstring, "what is
not covered"); that is the main known error of the copies/shared readings and it is why all three are reported.
"""
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graph_engine.margin_net import MarginNet                          # noqa: E402
from graph_engine.numeric_rules import extract_numeric_claims          # noqa: E402
from graph_engine.provenance_rules import (extract_provenance,         # noqa: E402
                                           provenance_abstentions, provenance_sources)

from _data import DATA
OUT = Path(__file__).with_name("e26_results.json")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = "hunt3fold_engine_e26"
EMAIL = os.environ.get("NCBI_EMAIL", "")
DELAY = 0.4                     # NCBI: at most 3 requests per second without an API key
BATCH = 200

QUERIES = {
    "bp_target": ("blood pressure target systolic mortality randomized", 2500),
    "bp_intensive_meta": ("intensive blood pressure control meta-analysis", 1500),
    "bp_ratio": ('"blood pressure"[tiab] AND (intensive[tiab] OR target[tiab] OR lowering[tiab]) AND '
                 '("hazard ratio"[tiab] OR "relative risk"[tiab] OR "risk ratio"[tiab])', 1600),
    "bp_outcomes": ('hypertension[tiab] AND ("all-cause mortality"[tiab] OR "cardiovascular events"[tiab]) AND '
                    '("hazard ratio"[tiab] OR "95% CI"[tiab])', 2000),
    # the copy-heavy part of the literature: abstracts that re-analyse one trial, and the trial most re-analysed
    "bp_secondary": ('("secondary analysis"[tiab] OR "post hoc analysis"[tiab] OR "pooled analysis"[tiab]) AND '
                     '"blood pressure"[tiab]', 1200),
    "sprint": ('SPRINT[tiab] AND "blood pressure"[tiab]', 900),
    "intensive_hr": ('("intensive blood pressure"[tiab] OR "blood pressure target"[tiab]) AND '
                     '("hazard ratio"[tiab] OR "95% CI"[tiab])', 500),
}

# the quantity a claim reports is decided from its quantity phrase PLUS the 120 characters before it (abstracts write
# "...death (hazard ratio 0.75, 95% CI ...)", so the endpoint is often outside the phrase numeric_rules keeps) —
# the grouping of surface phrases into one quantity is the caller's job, as in e23.
TARGETS = {
    "mortality_hr": re.compile(r"mortalit|death|surviv", re.I),
    "cv_event_hr": re.compile(r"cardiovascular|cardiac|stroke|myocardial|major adverse|\bCVD\b|\bMACE\b", re.I),
}
VALUE_WINDOW, SIGMA_MAX, CONTEXT = (0.3, 2.0), 0.6, 160
# Surface normalisation before numeric_rules, measured on this corpus (see `numeric_channel_gap`): numeric_rules' CI
# pattern wants the value adjacent to the interval ("0.75 (95% CI 0.64-0.89)"), medicine writes a separator
# ("0.75, 95% CI 0.64-0.89", "HR 0.75; 95% CI, 0.64 to 0.89"). Dropping that ONE separator is a surface edit, not a
# second extractor; it lifts the CI mentions read from 11 % to 66 %. The real fix belongs in numeric_rules.
_CI_SEP = (re.compile(r"([\d.])\s*[,;]\s*(\(?\s*95\s*%\s*(?:CI|confidence interval))", re.I),
           re.compile(r"(95\s*%\s*(?:CI|confidence interval))\s*[,:=]\s*", re.I))


def normalise_ci(text: str) -> str:
    return _CI_SEP[1].sub(r"\1 ", _CI_SEP[0].sub(r"\1 \2", text))


_RATIO_WORD = re.compile(r"hazard ratio|risk ratio|rate ratio|relative risk|odds ratio|\bHR\b|\baHR\b|\bRR\b", re.I)
# the headline group is the treatment contrast the owner asked for, not "any HR in a BP paper"
_ARM_WORD = re.compile(r"intensive|tight|strict|standard (?:treatment|care|target|therapy)|less intensive|usual care"
                       r"|randomi[sz]ed to|treatment (?:group|arm)|active treatment|antihypertensive", re.I)
_CI_TEXT = re.compile(r"95\s*%\s*(?:CI|confidence interval)", re.I)
RHO_SHARED = 0.5


# -- corpus --------------------------------------------------------------------------------------
def _get(endpoint: str, params: dict, data: dict | None = None) -> str:
    p = {"db": "pubmed", "tool": TOOL, **({"email": EMAIL} if EMAIL else {}), **params}
    url = EUTILS + endpoint + "?" + urllib.parse.urlencode(p)
    body = urllib.parse.urlencode(data).encode() if data else None
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=120).read().decode("utf-8", "replace")
        except Exception as e:                                          # noqa: BLE001
            print(f"  {endpoint} failed ({e}), retry {attempt}", flush=True)
            time.sleep(2 + 3 * attempt)
    return ""


def _text(node) -> str:
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip() if node is not None else ""


def _parse(xml: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        print("  parse error:", e, flush=True)
        return out
    for art in root.iter("PubmedArticle"):
        pmid = _text(art.find(".//MedlineCitation/PMID"))
        parts = []
        for ab in art.iter("AbstractText"):
            lab = ab.get("Label")
            t = _text(ab)
            if t:
                parts.append(f"{lab}: {t}" if lab else t)
        if not pmid or not parts:
            continue
        out.append({"pmid": pmid, "title": _text(art.find(".//ArticleTitle")), "abstract": " ".join(parts),
                    "journal": _text(art.find(".//Journal/ISOAbbreviation")),
                    "year": _text(art.find(".//JournalIssue/PubDate/Year")) or _text(art.find(".//PubDate/MedlineDate"))[:4],
                    "types": [_text(t) for t in art.iter("PublicationType")]})
    return out


def fetch(topic: str, term: str, want: int) -> list[dict]:
    """Cache-first. esearch for the PMIDs, efetch the abstracts in batches of 200, 0.4 s apart."""
    cache = DATA / f"pubmed_{topic}.jsonl"
    recs, have = [], set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            if r["pmid"] not in have:
                have.add(r["pmid"]); recs.append(r)
    if len(recs) >= want:
        return recs
    ids: list[str] = []
    while len(ids) < want:
        xml = _get("esearch.fcgi", {"term": term, "retstart": len(ids), "retmax": min(2000, want - len(ids)),
                                    "sort": "relevance"})
        got = re.findall(r"<Id>(\d+)</Id>", xml)
        time.sleep(DELAY)
        if not got:
            break
        ids += got
    ids = [i for i in ids if i not in have]
    print(f"  {topic}: {len(ids)} new pmids", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(ids), BATCH):
        xml = _get("efetch.fcgi", {"retmode": "xml", "rettype": "abstract"}, data={"id": ",".join(ids[i:i + BATCH])})
        for r in _parse(xml):
            if r["pmid"] not in have:
                have.add(r["pmid"]); r["topic"] = topic; recs.append(r)
        time.sleep(DELAY)
        if i % (BATCH * 5) == 0:
            print(f"  {topic}: {len(recs)} abstracts", flush=True)
        with cache.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    with cache.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return recs


def corpus() -> list[dict]:
    recs, seen = [], set()
    for topic, (term, want) in QUERIES.items():
        for r in fetch(topic, term, want):
            if r["pmid"] not in seen:
                seen.add(r["pmid"]); r.setdefault("topic", topic); recs.append(r)
    return recs


# -- the one quantity ----------------------------------------------------------------------------
def quantity_reports(recs):
    """{name: {pmid: claim}} — one report per abstract per quantity (the one with the smallest sigma).
    A claim qualifies when it is unitless, carries a stated CI, its value is a plausible ratio, and its phrase+context
    names a ratio AND one endpoint. A claim that matches both endpoints goes to the first one that matches.
    Each endpoint has two groups: `<name>` (any ratio for that endpoint in this BP corpus — a mixed bag of treatment
    effects and risk-factor associations) and `<name>_treatment_contrast` (the context also names a treatment arm:
    intensive / standard / usual care / randomised to …), which is the quantity the owner asked for."""
    lo, hi = VALUE_WINDOW
    groups = {k: {} for k in list(TARGETS) + [f"{k}_treatment_contrast" for k in TARGETS]}
    for r in recs:
        a = normalise_ci(r["abstract"])
        for c in extract_numeric_claims(a):
            if c["unit"] is not None or "ci" not in c["flags"] or "range_uniform" in c["flags"]:
                continue                                    # a ratio is unitless and must come with a CI
            if not (lo <= c["value"] <= hi and 0 < c["sigma"] < SIGMA_MAX):
                continue
            ctx = a[max(0, c["span"][0] - CONTEXT):c["span"][0]] + " " + c["quantity"]
            if not _RATIO_WORD.search(ctx):
                continue
            for name, pat in TARGETS.items():
                if pat.search(ctx):
                    names = [name] + ([f"{name}_treatment_contrast"] if _ARM_WORD.search(ctx) else [])
                    for nm in names:
                        g = groups[nm]
                        if r["pmid"] not in g or c["sigma"] < g[r["pmid"]]["sigma"]:
                            g[r["pmid"]] = c
                    break
    return groups


def numeric_channel_gap(recs):
    """How much of the corpus's CI reporting numeric_rules reads. Its CI pattern is the hep-ex surface form
    ("0.75 (95% CI 0.64-0.89)"); the common medical forms with a separator ("0.75, 95% CI 0.64-0.89",
    "HR 0.75; 95% CI 0.64 to 0.89") are NOT matched. Measured, not asserted."""
    with_ci = sum(1 for r in recs if _CI_TEXT.search(r["abstract"]))
    n_ci_mentions = sum(len(_CI_TEXT.findall(r["abstract"])) for r in recs)
    read = sum(1 for r in recs for c in extract_numeric_claims(r["abstract"]) if "ci" in c["flags"])
    read_n = sum(1 for r in recs for c in extract_numeric_claims(normalise_ci(r["abstract"])) if "ci" in c["flags"])
    return {"abstracts_mentioning_95pct_CI": with_ci, "CI_mentions_total": n_ci_mentions,
            "CI_claims_read_raw": read, "CI_claims_read_after_separator_normalisation": read_n,
            "fraction_of_CI_mentions_read_raw": round(read / max(n_ci_mentions, 1), 3),
            "fraction_of_CI_mentions_read_normalised": round(read_n / max(n_ci_mentions, 1), 3),
            "note": "numeric_rules' CI regex needs the value adjacent to the interval; the medical forms with a "
                    "comma or semicolon before '95% CI' are missed. This is a limit of the numeric channel found by "
                    "this corpus, not of provenance_rules."}


def run_net(items, mode, prov):
    """items = [(pmid, claim)]; prov = {pmid: [provenance records]}. mode: independent | copies | shared."""
    net = MarginNet()
    reports, rooted = [], 0
    for pmid, claim in items:
        st = [p for p in prov.get(pmid, []) if p["kind"] in ("copy", "derived", "shared")]
        if mode != "independent" and st:
            kinds = ("copy", "derived", "shared")
            recs = provenance_sources(pmid, [dict(p, kind="copy" if mode == "copies" else "derived") for p in st],
                                      rho=RHO_SHARED, kinds=kinds)
            if recs:
                net.add_sources(recs); rooted += 1
        reports.append({"margin": claim["value"], "sigma": claim["sigma"], "sources": [pmid]})
    net.add_edge("q", ["q"], reports)
    e = net.estimate("q")
    return {"m_hat": round(e.m, 4), "s": round(e.s, 4), "n_eff": round(e.n_eff, 2), "chi2_Q": round(e.q, 1),
            "dof": e.dof, "p_agree": float(f"{e.p_agree:.3g}"), "kind": e.kind, "n_reports": e.n_reports,
            "n_with_declared_lineage": rooted}


# -- hand check (25 random provenance extractions, seed 26, read one by one by the author) ---------
# key: "<pmid>:<target>"; verdict right | wrong. Abstracts sampled where the extractor produced NOTHING but the text
# does declare a traceable source are listed under ABSTAINED_SHOULD_HAVE by pmid.
HAND_VERDICTS = {
 "29564978:RAAS": ("wrong", "'based on the median daily equivalent dose of RAAS blocking drugs' — RAAS is a "
                            "physiological system, not a source; the abstract's real root (HALT PKD Study A) is a "
                            "two-word trial name and is not caught either"),
 "42238642:SPRINT": ("right", "'Using data from the Systolic Blood Pressure Intervention Trial (SPRINT)'"),
 "40251275:KDIGO guideline": ("right", "'following the 2021 KDIGO guidelines versus the 2012 KDIGO guidelines'"),
 "38380494:BOX": ("right", "'secondary analysis of the BOX trial'"),
 "18931896:EUROPA": ("right", "'the EUROPA study' — a mention, kind cites, no lineage edge"),
 "38063018:CABL": ("right", "'the CABL study' (Cardiovascular Abnormalities and Brain Lesions), cites"),
 "41779383:D2EFT": ("right", "'Secondary analysis of the D2EFT trial'"),
 "42541580:NUPRESS": ("right", "'data from the Brazilian multicentre trial (NUPRESS)'"),
 "25145522:JNC-8": ("right", "'as recommended by the JNC-8 panel' — the guideline body with its edition"),
 "19757004:FIELD": ("right", "'The FIELD study', cites"),
 "27456518:SPRINT": ("right", "'the SPRINT trial', cites"),
 "30789921:SSCP": ("wrong", "'The SSCP group' is a study ARM (structured secondary cardiovascular prevention "
                             "programme) of this paper's own cohort, not a source it copies"),
 "35140128:LUST": ("right", "'post hoc analysis of the LUST trial'"),
 "41911941:REPRIEVE": ("right", "'The REPRIEVE trial', cites"),
 "32067836:SPRINT": ("right", "'within the SPRINT trial including all participants ...'"),
 "30170330:NET": ("wrong", "the source is the PE-NET cohort ('the Pre-Eclampsia New Emerging Team [PE-NET] "
                            "cohort'); the bracketed name is truncated to NET"),
 "35489133:MAPT": ("right", "'data from the Multidomain Alzheimer Preventive Trial (MAPT)'"),
 "38150260:STRONG-HF": ("right", "'secondary analysis of the STRONG-HF randomized clinical trial'"),
 "40845203:SPRINT": ("right", "'post hoc analysis of the SPRINT trial'"),
 "29044764:SPRINT": ("right", "'post hoc analysis of the SPRINT.'"),
 "40270333:SPRINT": ("right", "'using data from the Systolic Blood Pressure Intervention Trial (SPRINT)'"),
 "31412746:ACC/AHA guideline": ("right", "'based on the 2017 ... (ACC/AHA) High Blood Pressure Guidelines'"),
 "27289122:ACCORD": ("right", "'the ACCORD trial', cites"),
 "31302044:MESA": ("right", "'the MESA cohort', cites"),
 "41684662:HVC": ("wrong", "'based on the presence of risk factors ... (either PH or HVC)' — HVC is heart valve "
                           "calcification, a risk factor of this paper's own cohort, not a source"),
}
ABSTAINED_SHOULD_HAVE = {
 "33448580": "the abstract IS a SPRINT secondary analysis ('SPRINT was a randomized clinical trial in which ...') "
             "but never says so with a provenance marker — no marker, no record",
 "29449941": "'Data from the Africa Middle East Cardiovascular Epidemiological (ACE) study were used' — ACE is on "
             "the NOT_A_TRIAL list (ACE inhibitors), so the one nameable root in the abstract is suppressed",
 "40469023": "'global health care data from the TriNetX network' — a mixed-case source name and 'network' is not a "
             "trial word; neither rule reaches it",
}


def hand_sample(recs, prov, n=25, seed=26):
    rng = random.Random(seed)
    pool = [r for r in recs if prov.get(r["pmid"])]
    picked = rng.sample(pool, min(n, len(pool)))
    out = []
    for r in picked:
        p = rng.choice(prov[r["pmid"]])
        out.append({"pmid": r["pmid"], "key": f"{r['pmid']}:{p['target']}", "record": p,
                    "span_text": r["abstract"][p["span"][0]:p["span"][1]], "abstract": r["abstract"][:1200]})
    return out


def abstention_sample(recs, prov, n=15, seed=126):
    rng = random.Random(seed)
    pool = [r for r in recs if not prov.get(r["pmid"])]
    return [{"pmid": r["pmid"], "abstract": r["abstract"][:1200],
             "markers": [a["marker"] for a in provenance_abstentions(r["abstract"])]}
            for r in rng.sample(pool, min(n, len(pool)))]


def main():
    recs = corpus()
    print(f"corpus {len(recs)} abstracts", flush=True)
    prov = {r["pmid"]: extract_provenance(r["abstract"]) for r in recs}
    absts = {r["pmid"]: provenance_abstentions(r["abstract"]) for r in recs}

    n = len(recs)
    kinds = Counter(p["kind"] for ps in prov.values() for p in ps)
    roots = Counter(p["target"] for ps in prov.values() for p in ps)
    lineage_roots = Counter(p["target"] for ps in prov.values() for p in ps if p["kind"] != "cites")
    res = {
        "corpus": {"n_abstracts": n, "queries": {k: v[0] for k, v in QUERIES.items()},
                   "source": "NCBI E-utilities, db=pubmed (esearch + efetch), cached outside the repo",
                   "years": sorted({r["year"] for r in recs if r["year"]})[:1]
                            + sorted({r["year"] for r in recs if r["year"]})[-1:]},
        "coverage": {
            "fraction_with_at_least_one_provenance_statement": round(sum(bool(p) for p in prov.values()) / n, 3),
            "fraction_with_a_lineage_statement_copy_derived_shared":
                round(sum(any(x["kind"] != "cites" for x in p) for p in prov.values()) / n, 3),
            "fraction_with_a_marker_but_no_nameable_source":
                round(sum(bool(absts[k]) and not prov[k] for k in prov) / n, 3),
            "statements_total": sum(len(p) for p in prov.values()),
            "statements_per_abstract_mean": round(sum(len(p) for p in prov.values()) / n, 2),
            "by_kind": dict(kinds), "abstention_markers_total": sum(len(a) for a in absts.values()),
            "distinct_named_roots": len(roots)},
        "top_roots_all_kinds": [[t, c] for t, c in roots.most_common(15)],
        "top_roots_lineage_only": [[t, c] for t, c in lineage_roots.most_common(15)],
        "quantities": {},
    }

    res["numeric_channel_gap"] = numeric_channel_gap(recs)
    groups = quantity_reports(recs)
    for name, g in groups.items():
        items = sorted(g.items())
        if len(items) < 3:
            res["quantities"][name] = {"n_reports": len(items)}
            continue
        declared = Counter(p["target"] for pmid, _ in items for p in prov[pmid] if p["kind"] != "cites")
        res["quantities"][name] = {
            "n_reports": len(items),
            "values_min_max": [round(min(c["value"] for _, c in items), 3), round(max(c["value"] for _, c in items), 3)],
            "median": round(sorted(c["value"] for _, c in items)[len(items) // 2], 3),
            "declared_roots_in_this_group": dict(declared.most_common(10)),
            "independent": run_net(items, "independent", prov),
            "copies": run_net(items, "copies", prov),
            f"shared_rho{RHO_SHARED}": run_net(items, "shared", prov),
        }

    sample = hand_sample(recs, prov)
    counts = Counter()
    for it in sample:
        counts[HAND_VERDICTS.get(it["key"], ("unjudged",))[0]] += 1
    ab_sample = abstention_sample(recs, prov)
    for it in ab_sample:
        counts["abstained_should_have" if it["pmid"] in ABSTAINED_SHOULD_HAVE else "correct_abstention"] += 1
    res["hand_check"] = {"n_extractions_read": len(sample), "n_abstentions_read": len(ab_sample), "seed": 26,
                         "counts": dict(counts),
                         "verdicts": {k: list(v) for k, v in HAND_VERDICTS.items()},
                         "abstained_should_have": {k: v for k, v in ABSTAINED_SHOULD_HAVE.items()},
                         "note": "25 extractions (one random record per sampled abstract) and 15 abstracts where the "
                                 "extractor produced nothing, all read by the author; 'unjudged' must stay 0"}
    res["hand_check_items"] = sample
    res["hand_check_abstentions"] = ab_sample
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("hand_check_")}, indent=1))


if __name__ == "__main__":
    main()
