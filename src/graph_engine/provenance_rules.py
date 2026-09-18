#!/usr/bin/env python3
"""
provenance_rules.py — deterministic extraction of PROVENANCE statements from English scientific text.

`numeric_rules` reads a number with an uncertainty out of a sentence; `margin_net` combines such numbers with a
lineage-aware GLS and asks with χ² whether they estimate one number. margin_net has to be TOLD the lineage
(`derives_from` = copy of a root, `shares={group: ρ}` = partially shared error, nothing = independent). In physics the
lineage is guessable from the collaboration name (e23). In medicine it is not: guidelines, reviews, editorials and
textbooks are copies of a few trials, so "many independent reports" is usually many copies of one root — and the
lineage is stated in prose, if at all. This module is that channel: text → provenance records, by regular expressions
only, no model, no learning, same answer every run.

A record is
    {"kind": "copy" | "derived" | "shared" | "cites", "target": "<named source>", "span": (a, b), "confidence": float}
and is emitted ONLY when a provenance marker fires AND a source can be NAMED in the same sentence after it.
Marker without a nameable target (e.g. "a meta-analysis of 42 randomised trials", "consistent with previous reports")
is an ABSTENTION: it is not returned by `extract_provenance`; `provenance_abstentions` returns those spans separately
so a caller can count them instead of guessing a root.

Mapping to margin_net's three lineage states (the caller chooses ρ; this module never does):
    copy    → add_sources([{"id": report, "derives_from": [target]}])   the report restates the target's conclusion;
                                                                        copies of one root do not narrow an estimate
                                                                        (N_eff counts the root once) and margin_net's
                                                                        copy check demands that they AGREE.
    derived → add_sources([{"id": report, "shares": {target: ρ}}])      a new analysis of the target's data (pooled /
    shared  → add_sources([{"id": report, "shares": {target: ρ}}])      meta / secondary / same cohort): its error is
                                                                        partly the target's, not all of it. `derived`
                                                                        (re-analysis of results) and `shared` (same
                                                                        underlying data) differ only in the ρ a caller
                                                                        would pick — both are the `shares` state.
    cites   → NO lineage edge.                                          Agreement with a named source is not the same
                                                                        as being computed from it.
`provenance_sources(report_id, statements, rho=...)` builds exactly those `add_sources` records.

Markers covered (case-insensitive, the target is searched to the right inside the same sentence, ≤ WINDOW chars):
    copy     "according to X", "as recommended by X", "following the X guideline(s)", "in accordance with X",
             "in line with the X recommendations", "as defined by X", "per the X guidelines", "adhering to X"
    derived  "meta-analysis of …", "pooled analysis of …", "systematic review of …", "we re-analysed / reanalyzed …",
             "a secondary analysis of …", "post hoc analysis of …", "subgroup analysis of …", "individual patient data
             from …"
    shared   "data from X", "using data from X", "participants (enrolled) in X", "in the X cohort/registry",
             "derived from X", "the X trial population"
    cites    "consistent with X", "in agreement with X", "as reported in/by X", "confirms the findings of X",
             "similar to that reported in X", and a bare named trial ("the SPRINT trial showed …")
A target is NAMED when the window contains either
    (a) a guideline body from `GUIDELINE_BODIES` (ESC, ESH, AHA, ACC, NICE, WHO, JNC, KDIGO, ADA, ISH, USPSTF, …),
        optionally slash-joined ("ESC/ESH"), normalised to "<BODY> guideline"; or
    (b) a trial/cohort acronym: an ALL-CAPS token (3–14 chars, digits/hyphen allowed, not in `NOT_A_TRIAL`) that has
        one of trial / study / cohort / registry / investigators / collaboration / group within two tokens, or that
        follows "the" directly after a derivation marker. Normalised to the bare acronym ("the SPRINT trial" → SPRINT).

Confidence is a fixed table, not a probability: 0.9 marker + acronym with an explicit trial/cohort word, 0.8 marker +
guideline body, 0.7 bare trial mention (kind `cites`), 0.6 acronym named by "the" alone after a marker. It orders
records; it is not calibrated against anything.

What is NOT covered, on purpose (abstentions, not silent guesses):
  * unnamed sources: "previous studies", "recent trials", "the literature", "our earlier work", "a meta-analysis of 12
    trials" with no acronym — no target, no record;
  * reference-list lineage ([12], "Smith et al. 2015"): author-year and bracket citations are not resolved to a root;
  * negated and contrastive provenance ("in contrast to SPRINT", "unlike the ACCORD trial") is read as `cites`, and
    "we did NOT use data from X" is not detected as a negation at all — a known false positive;
  * which of several named sources a given NUMBER comes from: the record names the abstract's sources, not the
    sentence-level attachment of one value;
  * the strength of a copy (a guideline that quotes one trial vs. one that weighs twenty) — that is ρ, the caller's;
  * non-English text, and trials whose name is not an acronym ("the Framingham Heart Study" is caught only through
    FRAMINGHAM-style caps, i.e. usually not).
Hand check of 25 random PubMed abstracts: see examples/engine_experiments/e26_results.json.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["extract_provenance", "provenance_abstentions", "provenance_sources",
           "GUIDELINE_BODIES", "NOT_A_TRIAL", "KIND_TO_LINEAGE", "WINDOW"]

WINDOW = 140          # characters after the marker searched for a named source

KIND_TO_LINEAGE = {"copy": "derives_from", "derived": "shares", "shared": "shares", "cites": None}

GUIDELINE_BODIES = ["ESC", "ESH", "AHA", "ACC", "ACP", "NICE", "WHO", "JNC", "KDIGO", "ADA", "ISH", "USPSTF",
                    "CHEP", "NHLBI", "NKF", "EAS", "ERS", "IDF", "SIGN", "AACE", "ASH", "ACCF", "HFSA", "CCS",
                    "JSH", "CHS", "ESO", "EASD", "AAFP", "NHS", "NIH", "FDA", "EMA", "ISHIB"]

# ALL-CAPS tokens that are not trial names (units, statistics, diseases, drug classes, common medical abbreviations)
NOT_A_TRIAL = {
    "BP", "SBP", "DBP", "MAP", "HR", "RR", "OR", "CI", "SD", "SE", "IQR", "HRS", "CVD", "CV", "CHD", "CAD", "CKD",
    "ESRD", "MI", "AMI", "HF", "CHF", "AF", "LVH", "LVEF", "EF", "TIA", "ACS", "PAD", "DM", "T2DM", "T1DM", "HTN",
    "LDL", "HDL", "BMI", "ECG", "EKG", "ACE", "ACEI", "ARB", "CCB", "NNT", "NNH", "ITT", "RCT", "RCTS", "USA",
    "US", "UK", "EU", "EUROPE", "AND", "OR", "THE", "FOR", "NOT", "ALL", "PLUS", "VS", "CKDEPI", "EGFR", "GFR",
    "CRP", "HBA1C", "HBA", "COVID", "SARS", "HIV", "AIDS", "COPD", "OSA", "BMJ", "JAMA", "NEJM", "PRISMA",
    "MOOSE", "GRADE", "PROSPERO", "MEDLINE", "EMBASE", "PUBMED", "CENTRAL", "CONSORT", "AGREE", "STROBE",
    "ICD", "WHOQOL", "MESH", "DOI", "PDF", "NA", "ID", "II", "III", "IV", "VI", "VII", "MD", "SMD", "PY",
    "ABPM", "HBPM", "SPRINTED", "MRI", "CT", "PET", "ICU", "ER", "GP", "QOL", "BSA", "NYHA", "KDOQI",
    # section labels of structured abstracts: all caps, next to the word "trial"/"study", and not a trial name
    "BACKGROUND", "METHODS", "METHOD", "RESULTS", "RESULT", "CONCLUSION", "CONCLUSIONS", "OBJECTIVE", "OBJECTIVES",
    "PURPOSE", "DESIGN", "SETTING", "PARTICIPANTS", "PATIENTS", "INTERVENTION", "INTERVENTIONS", "OUTCOME",
    "OUTCOMES", "MEASURES", "FUNDING", "REGISTRATION", "CLINICAL", "TRIAL", "TRIALS", "STUDY", "STUDIES",
    "IMPORTANCE", "CONTEXT", "LIMITATIONS", "IMPLICATIONS", "INTRODUCTION", "AIM", "AIMS", "DISCUSSION",
    "ABSTRACT", "KEYWORDS", "SUMMARY", "EVIDENCE", "REVIEW", "MAIN", "DATA", "SOURCES", "SELECTION", "ANALYSIS",
    "SYNTHESIS", "GOV", "NUMBER", "IDENTIFIER", "PROTOCOL", "ETHICS", "DISSEMINATION", "APPROVAL", "COHORT",
}

_STOP_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n")
_TRIAL_WORD = r"(?:trial|trials|study|studies|cohort|cohorts|registry|registries|investigators|collaboration|group|programme|program)"

# marker -> (kind, base confidence source). Each pattern's END is where the target search starts.
_MARKERS: list[tuple[str, re.Pattern]] = [
    ("copy", re.compile(r"\b(?:according to|as recommended by|as recommended in|as advised by|in accordance with"
                        r"|in compliance with|as defined by|as specified by|as per|per the|adhering to|adherent to"
                        r"|following the|based on the|guided by|in keeping with the recommendations of"
                        r"|as stated in|in line with the recommendations of|recommended by|followed the|follows the)\b", re.I)),
    ("derived", re.compile(r"\b(?:meta-?analys[ie]s of|pooled analys[ie]s of|pooled data from|systematic review of"
                           r"|we (?:re-?analy[sz]ed|re-?examined)|re-?analys[ie]s of|secondary analys[ie]s of"
                           r"|post[- ]hoc analys[ie]s of|subgroup analys[ie]s of|exploratory analys[ie]s of"
                           r"|individual (?:patient|participant) data (?:from|of)|network meta-?analys[ie]s of"
                           r"|an? analysis of (?:the )?(?:data from )?)", re.I)),
    ("shared", re.compile(r"\b(?:using data from|data (?:were |was )?(?:obtained |drawn |taken )?from"
                          r"|participants (?:enrolled )?in|patients (?:enrolled )?in|derived from"
                          r"|we used (?:the )?data of|within the)\b", re.I)),
    ("cites", re.compile(r"\b(?:consistent with|in agreement with|in contrast to|unlike|as reported in|as reported by"
                         r"|confirms the findings of|confirm the findings of|similar to (?:that |those )?reported in"
                         r"|comparable to (?:that |those )?(?:of|in)|in line with)\b", re.I)),
]

# markers whose kind depends on WHAT is named: a guideline is restated (copy), a trial's data is re-used (derived)
_AMBIVALENT = {"based on the", "guided by", "following the", "followed the", "follows the"}
_KIND_RANK = {"copy": 0, "derived": 1, "shared": 2, "cites": 3}     # one record per target: the strongest claim wins

_BODY_RE = re.compile(r"\b(" + "|".join(GUIDELINE_BODIES) + r")(?:\s*/\s*(?:" + "|".join(GUIDELINE_BODIES) + r"))*\b")
_ACRO = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{2,13}(?:-[A-Z0-9]{1,6})?)(?![A-Za-z0-9])")
# a bare named trial anywhere: "the SPRINT trial", "the ACCORD-BP study"
_BARE_TRIAL = re.compile(rf"\b[Tt]he\s+([A-Z][A-Z0-9]{{2,13}}(?:-[A-Z0-9]{{1,6}})?)\s+{_TRIAL_WORD}\b")


def _sentence_end(text: str, start: int) -> int:
    m = _STOP_SENT.search(text, start)
    return min(m.start() if m else len(text), start + WINDOW)


def _is_trial_token(tok: str) -> bool:
    return (tok not in NOT_A_TRIAL and tok not in GUIDELINE_BODIES and not tok.isdigit()
            and len(tok) >= 3 and any(c.isalpha() for c in tok))


def _named_targets(text: str, start: int, stop: int) -> list[tuple[str, float]]:
    """Every nameable source in text[start:stop]: guideline bodies and trial acronyms. [] = abstain."""
    win = text[start:stop]
    cands: list[tuple[int, str, float]] = []
    mb = _BODY_RE.search(win)
    if mb:
        body = re.sub(r"\s*/\s*", "/", mb.group(0))
        cands.append((mb.start(), f"{body} guideline", 0.8))
    for ma in _ACRO.finditer(win):
        tok = ma.group(1)
        if not _is_trial_token(tok):
            continue
        after = win[ma.end():ma.end() + 24]
        before = win[max(0, ma.start() - 24):ma.start()]
        near = re.match(rf"\W*(?:\w+\W+){{0,1}}{_TRIAL_WORD}\b", after, re.I) or \
            re.search(rf"\b{_TRIAL_WORD}\W+(?:\w+\W+){{0,1}}$", before, re.I)
        if near:
            cands.append((ma.start(), tok, 0.9))
        elif re.search(r"\b(?:the|of|from|in)\s*$", win[:ma.start()], re.I) and ma.start() <= 40:
            cands.append((ma.start(), tok, 0.6))
    best: dict[str, tuple[int, float]] = {}
    for pos, name, conf in cands:
        if name not in best or conf > best[name][1]:
            best[name] = (pos, conf)
    return [(n, c) for n, (p, c) in sorted(best.items(), key=lambda kv: kv[1][0])]


def _scan(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hits: list[dict[str, Any]] = []
    absts: list[dict[str, Any]] = []
    if not text:
        return hits, absts
    marks = sorted(((m.start(), m.end(), kind, m.group(0)) for kind, rx in _MARKERS for m in rx.finditer(text)))
    for i, (a, b, kind, surface) in enumerate(marks):
        nxt = next((x for x, _, _, _ in marks[i + 1:] if x > b), len(text))   # a later marker owns its own target
        stop = min(_sentence_end(text, b), nxt)
        got = _named_targets(text, b, stop)
        if not got:
            absts.append({"kind": kind, "target": None, "span": (a, b), "marker": surface, "confidence": 0.0})
            continue
        for target, conf in got:
            k = kind
            if surface.lower() in _AMBIVALENT:              # "based on the X" — a guideline is copied, a trial is used
                k = "copy" if target.endswith(" guideline") else "derived"
            hits.append({"kind": k, "target": target, "span": (a, stop), "confidence": conf})
    for m in _BARE_TRIAL.finditer(text):                    # "the SPRINT trial showed ..." — a mention, no lineage
        tok = m.group(1)
        if _is_trial_token(tok) and not any(a <= m.start() < b for a, b in (h["span"] for h in hits)):
            hits.append({"kind": "cites", "target": tok, "span": (m.start(), m.end()), "confidence": 0.7})
    hits.sort(key=lambda h: h["span"])
    absts.sort(key=lambda h: h["span"])
    return hits, absts


def extract_provenance(text: str, dedupe: bool = True) -> list[dict[str, Any]]:
    """Provenance statements with a NAMED source, left to right. See the module docstring.

    dedupe=True (default) keeps ONE record per target — the strongest kind (copy > derived > shared > cites), and
    within a kind the highest confidence — because a caller building lineage wants one edge per source, not one per
    sentence. dedupe=False returns every marker hit, for auditing.
    """
    hits, _ = _scan(text)
    if not dedupe:
        return hits
    best: dict[str, dict[str, Any]] = {}
    rank = lambda h: (_KIND_RANK[h["kind"]], -h["confidence"], h["span"])    # noqa: E731
    for h in hits:
        if h["target"] not in best or rank(h) < rank(best[h["target"]]):
            best[h["target"]] = h
    return sorted(best.values(), key=lambda h: h["span"])


def provenance_abstentions(text: str) -> list[dict[str, Any]]:
    """Marker fired, no source could be named: the honest count of 'provenance is claimed but not traceable'."""
    return _scan(text)[1]


def provenance_sources(report_id: str, statements: list[dict[str, Any]], rho: float = 0.5,
                       min_confidence: float = 0.0, kinds: tuple[str, ...] = ("copy", "derived", "shared"),
                       ) -> list[dict[str, Any]]:
    """margin_net.add_sources records for one report, from its provenance statements.

    copy → derives_from (the report IS the root); derived/shared → shares {target: rho} (the caller owns rho);
    cites → nothing. Returns [] when nothing qualifies, i.e. the report stays its own independent root.
    """
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho = {rho} must be in (0, 1) — margin_net rejects 0 and 1")
    parents, shares = [], {}
    for st in statements:
        if st["kind"] not in kinds or st["confidence"] < min_confidence or not st.get("target"):
            continue
        if KIND_TO_LINEAGE[st["kind"]] == "derives_from":
            parents.append(st["target"])
        elif KIND_TO_LINEAGE[st["kind"]] == "shares":
            shares[st["target"]] = rho
    if not parents and not shares:
        return []
    rec: dict[str, Any] = {"id": report_id}
    if parents:
        rec["derives_from"] = sorted(set(parents))
        shares = {g: r for g, r in shares.items() if g not in rec["derives_from"]}
    if shares:
        if len(shares) > 1:                                 # margin_net caps the total shared fraction at 0.999
            shares = {g: rho / len(shares) for g in shares}
        rec["shares"] = shares
    return [rec]
