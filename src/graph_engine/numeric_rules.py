#!/usr/bin/env python3
"""
numeric_rules.py — deterministic extraction of numeric claims WITH AN UNCERTAINTY from English scientific text.

typed_extraction reads mechanism atoms and signs out of a sentence; nothing in the package read NUMBERS out of one.
margin_net combines reports m ± σ with lineage-aware GLS and tests, with χ², whether they estimate one number — but it
has to be handed m and σ. This module is that channel: text → {"value", "sigma", "unit", "quantity", "span", "flags"},
by regular expressions only, no model, no learning, same answer every run.

A record is emitted ONLY when the text states an uncertainty and a quantity phrase is found. What is covered:

  v ± s                     "172.52 ± 0.33 GeV"      also  +/-  , +- , \\pm
  several terms             "172.5 ± 0.7 (stat) ± 1.0 (syst)"  → σ = sqrt(Σ σ_i²), flag `quadrature`
  asymmetric                "172.5 ^{+0.8}_{-0.9}", "172.5 +0.8 -0.9"  → σ = max(a, b), flag `asymmetric`
                            (the two are not the same statement; max is the conservative reading — hence the flag)
  power of ten              "(1.2 ± 0.3) × 10^{-5}", "1.2e-5 ± 0.3e-5", trailing "× 10^-9"  → flag `power_of_ten`
  confidence interval       "1.34 (95% CI 1.10–1.63)", "95% CI [1.10, 1.63]"  → σ = (hi − lo) / (2 · 1.96),
                            value = the stated central value if there is one, else the midpoint; flag `ci`
                            "12.0 ± 1.5 (95% CI)" reads the ± as a half-width: σ = 1.5 / 1.96, flag `ci_halfwidth`
  range                     "3.2–4.8 GeV", "3.2 to 4.8 ms"  → a uniform: value = midpoint, σ = (w − v) / (2√3),
                            flag `range_uniform` — a range is not a measurement report, the flag says so
  LaTeX                     abstracts are LaTeX source: "$m_t = 172.95 \\pm 0.53$ GeV", "$176.1\\pm 5.1 (stat.) \\gevcc$",
                            "170.7\\pm0.3~(\\text{stat.})" — $ ~ \\, \\; { } are skipped between the pieces, the unit may be
                            a macro (\\GeV, \\gevcc, \\GeVc2, \\TeV, \\MeV), and the quantity phrase is stripped of markup
                            ("$m_{top}$" → "m_top"). "177.8+-4.5/5.0" is read as an asymmetric ±.
  units                     eV/keV/MeV/GeV/TeV (also "GeV/c^2") → GeV;  ps…s → s;  nm…km → m;  % → fraction of 1.
                            value and σ are converted to the base unit; `unit` is the base unit, None if none was given
                            (flag `no_unit`). Anything outside the table is kept verbatim in `unit` with flag
                            `unknown_unit` and is NOT converted.
  quantity                  the nearest phrase before the number, inside the same sentence and within
                            `WINDOW` characters: connectors ("of", "=", "is", "measured to be", …) are stripped and up
                            to `MAX_WORDS` words are kept — "m_t = 172.5 ± 0.3 GeV" → "m_t",
                            "the top quark mass of 172.5 ± 0.3 GeV" → "the top quark mass".

What is NOT covered, on purpose (these are abstentions, not silent guesses):
  * numbers without a stated uncertainty ("a 125 GeV boson") — nothing to give margin_net;
  * upper/lower limits ("< 1.2 × 10⁻⁹ at 95% CL"), significances ("5.1σ"), p-values, ±-free "order of" statements;
  * correlated-uncertainty structure: several terms are added in quadrature, i.e. assumed independent;
  * asymmetric intervals are collapsed to one σ (flagged), no skew-normal;
  * the quantity is the *surface phrase*, not a resolved concept: "the mass" and "m_t" are different strings.
    Grouping synonyms is the caller's job (see examples/engine_experiments/e23_numeric_conf_hepex.py);
  * when no quantity phrase survives the strip inside the window, the record is DROPPED (abstain), never guessed;
  * per-mille, dB, compound units (GeV⁻², cm²/s), dates, version numbers, citation years.

Author's check: 30 random extractions from arXiv hep-ex abstracts were read by hand, see e23_results.json.
"""
from __future__ import annotations

import math
import re
from typing import Any

__all__ = ["extract_numeric_claims", "normalize_unit", "UNITS", "WINDOW", "MAX_WORDS"]

WINDOW = 90          # characters before the number searched for the quantity phrase
MAX_WORDS = 8        # longest quantity phrase kept

# -- units ---------------------------------------------------------------------------------------
# surface form -> (factor to base, base unit)
UNITS: dict[str, tuple[float, str]] = {
    "eV": (1e-9, "GeV"), "keV": (1e-6, "GeV"), "MeV": (1e-3, "GeV"), "GeV": (1.0, "GeV"), "TeV": (1e3, "GeV"),
    "ps": (1e-12, "s"), "ns": (1e-9, "s"), "us": (1e-6, "s"), "µs": (1e-6, "s"), "μs": (1e-6, "s"),
    "ms": (1e-3, "s"), "s": (1.0, "s"),
    "nm": (1e-9, "m"), "um": (1e-6, "m"), "µm": (1e-6, "m"), "μm": (1e-6, "m"),
    "mm": (1e-3, "m"), "cm": (1e-2, "m"), "m": (1.0, "m"), "km": (1e3, "m"),
    "%": (0.01, "1"),
}
def normalize_unit(value: float, sigma: float, unit: str | None) -> tuple[float, float, str | None, list[str]]:
    """(value, sigma, unit) in the base unit of the table; unknown units are left alone and flagged."""
    if unit is None:
        return value, sigma, None, ["no_unit"]
    u = unit.strip()
    u = re.sub(r"\s*/\s*c\s*(\^\s*\{?\s*2\s*\}?|\*\*2|²)?$", "", u)  # GeV/c^2 is a mass in GeV
    if u not in UNITS:
        return value, sigma, unit, ["unknown_unit"]
    f, base = UNITS[u]
    return value * f, sigma * f, base, []


# -- numbers -------------------------------------------------------------------------------------
_N = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SN = r"[-+−]?" + _N
_PM = r"(?:±|\+/-|\+-|\\pm)"
# abstracts are LaTeX: $ ~ \, \; {} around the numbers must not break the match
_GAP = r"(?:[\s~$]|\\[,;!:>]|\\quad|\\qquad|\{\}|(?<=[\s~$])\{|\}(?=[\s~$,.;)]))*"
_LABEL_BODY = r"[A-Za-z][A-Za-z0-9.+\-\s]{0,18}"
_LAB = (rf"(?:{_GAP}(?:"
        rf"\(\s*(?:\\[A-Za-z]+\s*)?\{{?\s*{_LABEL_BODY}\}}?\s*\)"      # (stat), (\text{{stat.}}), (\rm{{stat}})
        # \rm{(stat+JSF)} — but \mathrm{GeV} is the UNIT, not a label, so it is excluded here
        rf"|(?!\\[A-Za-z]+\s*\{{\s*(?:[TGMk]?eV|ps|ns|ms|s|mb|pb|fb|nb|[num]?m|%)\s*\}})"
        rf"\\[A-Za-z]+\s*\{{\(?\s*{_LABEL_BODY}\)?\}}"
        rf"|_\s*\{{\s*(?:\\[A-Za-z]+\s*)?(?:\{{\s*{_LABEL_BODY}\}}|{_LABEL_BODY})\s*\}}"   # _{{\rm stat}}, _{{\mathrm{{stat.}}}}
        rf"|_\s*(?:stat|syst|sys)\b"
        rf"|\b(?:stat|syst|sys|lumi|theo|model|tot|recoil)\.?))?")
_POW = (rf"(?:{_GAP}(?:×|x|\*|\\times|\btimes\b){_GAP}10{_GAP}(?:\^|\*\*)?\s*\{{?\s*(?P<e>[-+−]?\d+)\s*\}}?)")

_TERM = re.compile(
    rf"{_GAP}(?:"
    rf"{_PM}{_GAP}(?P<s>{_N})(?:\s*/\s*(?P<s2>{_N}))?"
    rf"|\^\s*\{{?\s*\+\s*(?P<a>{_N})\s*\}}?\s*_\s*\{{?\s*[-−]\s*(?P<b>{_N})\s*\}}?"
    rf"|\+\s*(?P<a2>{_N})\s*[-−]\s*(?P<b2>{_N})\b"
    rf"){_LAB}"
)
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
# a unit token right after the number: one from the table, or any short symbol-like token that is not an English word
_NOT_A_UNIT = {"and", "or", "of", "in", "at", "for", "with", "was", "were", "is", "are", "the", "a", "an", "to",
               "on", "by", "from", "per", "as", "we", "it", "this", "that", "times", "x", "e", "data", "events",
               "times.", "sigma", "standard", "deviations", "each", "over", "than", "using", "which", "while"}
_UNIT_RE = re.compile(
    rf"{_GAP}(?:(?P<u>{_UNIT_ALT})|(?P<g>[A-Za-zµμΩ][A-Za-z0-9µμΩ^{{}}²³]{{0,5}}"
    rf"(?:\s*/\s*[A-Za-z0-9^{{}}²³]{{1,6}})?))"
    rf"(?P<c>\s*/\s*c\s*(?:\^\s*\{{?\s*2\s*\}}?|\*\*2|²)?)?(?![A-Za-z0-9])")


_VALUE = re.compile(rf"(?<![\w.]){_SN}")
_POW_RE = re.compile(_POW)
_CI_TAG = re.compile(r"\s*\(\s*(?:\d{2}(?:\.\d+)?\s*%\s*)?(?:CI|C\.I\.|confidence interval)\s*\)", re.I)
_CI = re.compile(
    rf"(?:(?P<v>{_SN})\s*)?[\(\[]?\s*(?:at\s+)?(?P<lvl>\d{{2}}(?:\.\d+)?)\s*%\s*"
    rf"(?:CI|C\.I\.|confidence interval)\s*[:=]?\s*[\[\(]?\s*(?P<lo>{_SN})\s*(?:,|–|—|-|\bto\b)\s*"
    rf"(?P<hi>{_SN})\s*[\]\)]?", re.I)
_RANGE = re.compile(rf"(?<![\w.])(?P<lo>{_N})\s*(?:–|—|\s+to\s+|-)\s*(?P<hi>{_N})")

_Z = {1.0: 0.0, 68.0: 1.0, 90.0: 1.6448536269514722, 95.0: 1.959963984540054, 99.0: 2.5758293035489004}


def _f(x: str) -> float:
    return float(x.replace("−", "-"))


def _z_for(level: float) -> float:
    return min(_Z.items(), key=lambda kv: abs(kv[0] - level))[1] or 1.959963984540054


# -- quantity phrase -----------------------------------------------------------------------------
_CONNECT = re.compile(
    r"(?:\s*(?:=|:|≈|≃|~|<|>|\bis\b|\bare\b|\bwas\b|\bwere\b|\bof\b|\bto\s+be\b|\bat\b|\babout\b"
    r"|\bapproximately\b|\baround\b|\bequal\s+to\b|\bequals\b|\byields?\b|\bgives?\b|\bfound\b|\bmeasured\b"
    r"|\bdetermined\b|\bobtained\b|\bestimated\b|\breported\b|\bvalue\b|\bwith\b|\bhas\b|\bbeing\b"
    r"|,|\(|\$|~|\\,|\\;|\{|\})\s*)+$",
    re.I)
_STOP = {"and", "or", "the", "a", "an", "we", "our", "this", "that", "it", "in", "for", "by", "from", "be",
         "on", "as", "than", "then", "both", "using", "here", "these", "those", "which", "where", "while",
         "respectively", "i", "ii", "e", "g"}
_CUT = {"is", "are", "was", "were", "be", "been", "being", "of", "to", "at", "in", "on", "for", "with",
        "from", "by", "as", "find", "finds", "found", "measure", "measured", "measurement", "measurements",
        "report", "reports", "reported", "obtain", "obtained", "determine", "determined", "give", "gives",
        "yield", "yields", "observe", "observed", "extract", "extracted", "present", "presents", "presented",
        "combine", "combined", "using", "and", "or", "that", "which", "than", "gives", "equal", "equals",
        "value", "values", "result", "results", "quoted", "set", "have", "has", "had", "show", "shows", "shown"}
_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(])|\n")


def _quantity(text: str, start: int) -> str | None:
    pre = text[max(0, start - WINDOW):start]
    parts = _SENT.split(pre)
    pre = parts[-1] if parts else pre
    pre = pre.split(";")[-1]
    prev = None
    while pre != prev:                                    # strip connectors repeatedly ("of the order of", "is at")
        prev = pre
        pre = _CONNECT.sub("", pre).rstrip()
    pre = re.sub(r"\s+\([^()]*\)\s*$", "", pre).rstrip()   # a trailing parenthetical is not the quantity
    if not pre or not re.search(r"[A-Za-z]", pre):
        return None
    pre = pre.split(",")[-1]
    m = re.search(r"[A-Za-z_$\\][\w$\\{}()^_/\-µμ→' ]*$", pre)
    if not m:
        return None
    words = m.group(0).split()
    cut = [i for i, w in enumerate(words) if w.lower().strip(".,") in _CUT]
    if cut:
        words = words[cut[-1] + 1:]                       # the phrase starts after the last verb / preposition
    words = words[-MAX_WORDS:]
    while words and re.fullmatch(r"[\W_]+", words[0]):
        words = words[1:]
    while words and words[0].lower() in _STOP and len(words) > 1:
        words = words[1:]
    if not words:
        return None
    phrase = _delatex(" ".join(words)).strip(" -^_")
    if not phrase or phrase.lower() in _STOP or not re.search(r"[A-Za-z]{2}|[A-Za-z]_", phrase):
        return None
    return phrase


def _delatex(s: str) -> str:
    """$m_{top}$ → m_top, $m_{t}^\\text{pole}$ → m_t^pole: the phrase is a name, its markup is not part of it."""
    s = re.sub(r"\\(?:text|mathrm|rm|mathit|it|ensuremath|bf)\b|\\[,;]", " ", s)
    s = s.replace("$", "").replace("\\", "")
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# -- the scan ------------------------------------------------------------------------------------
_MACRO_UNITS = {"gev": "GeV", "gevc2": "GeV", "gevcc": "GeV", "tev": "TeV", "mev": "MeV", "mevcc": "MeV",
                "mevc2": "MeV", "kev": "keV", "ev": "eV", "nano": None}
_MACRO_RE = re.compile(rf"{_GAP}\\(?P<m>[A-Za-z]+)(?:\{{\}})?(?![A-Za-z])")
# \mathrm{GeV}, \rm{MeV}, \text{ps}: the unit is the macro's argument
_MACRO_ARG_RE = re.compile(rf"{_GAP}\\(?:mathrm|rm|text|mathit|mbox|unit)\s*\{{\s*(?P<u>[A-Za-z%µμ]{{1,5}})\s*\}}")


def _unit_at(text: str, pos: int) -> tuple[str | None, int]:
    ma = _MACRO_ARG_RE.match(text, pos)                    # \mathrm{GeV}, \rm{MeV}
    if ma:
        u, end = ma.group("u"), ma.end()
        mc = re.match(r"\s*/\s*c\s*(?:\^\s*\{?\s*2\s*\}?|\u00b2)?", text[end:])
        return (u + ("/c^2" if mc else ""), end + (mc.end() if mc else 0))
    mm = _MACRO_RE.match(text, pos)                        # LaTeX macro units: \GeV, \gevcc, \GeVc2, \TeV
    if mm and _MACRO_UNITS.get(mm.group("m").lower()):
        return _MACRO_UNITS[mm.group("m").lower()], mm.end()
    m = _UNIT_RE.match(text, pos)
    if not m:
        return None, pos
    if m.group("u") is None:
        g = m.group("g").strip()
        if g.lower() in _NOT_A_UNIT or re.fullmatch(r"[A-Za-z]{4,}", g):   # a word, not a unit symbol
            return None, pos
        return g + (m.group("c") or ""), m.end()
    return m.group("u") + (m.group("c") or ""), m.end()


def _emit(text: str, start: int, end: int, value: float, sigma: float, unit: str | None,
          flags: list[str]) -> dict[str, Any] | None:
    q = _quantity(text, start)
    if q is None or sigma <= 0 or not math.isfinite(sigma) or not math.isfinite(value):
        return None
    v, s, u, uf = normalize_unit(value, sigma, unit)
    return {"value": v, "sigma": s, "unit": u, "quantity": q, "span": (start, end),
            "flags": sorted(set(flags) | set(uf))}


def extract_numeric_claims(text: str) -> list[dict[str, Any]]:
    """All numeric claims with an uncertainty in `text`, left to right, non-overlapping. See the module docstring."""
    if not text:
        return []
    text = text.replace(" ", " ")
    taken: list[tuple[int, int]] = []
    out: list[dict[str, Any]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    # 1. confidence intervals (they contain ± -free number pairs and must be claimed first)
    for m in _CI.finditer(text):
        lo, hi, lvl = _f(m.group("lo")), _f(m.group("hi")), float(m.group("lvl"))
        if hi <= lo:
            continue
        sigma = (hi - lo) / (2 * _z_for(lvl))
        value = _f(m.group("v")) if m.group("v") else (lo + hi) / 2
        unit, end = _unit_at(text, m.end())
        if not free(m.start(), end):
            continue
        rec = _emit(text, m.start(), end, value, sigma, unit, ["ci"])
        if rec:
            out.append(rec); taken.append((m.start(), end))

    # 2. value ± term(s) [× 10^e] [unit]
    for mv in _VALUE.finditer(text):
        terms, pos, flags = [], mv.end(), []
        while True:
            mt = _TERM.match(text, pos)
            if not mt:
                break
            if mt.group("s"):
                if mt.group("s2"):                                  # "+-4.5/5.0" = +4.5 / -5.0
                    terms.append(max(_f(mt.group("s")), _f(mt.group("s2")))); flags.append("asymmetric")
                else:
                    terms.append(_f(mt.group("s")))
            else:
                a, b = (mt.group("a"), mt.group("b")) if mt.group("a") else (mt.group("a2"), mt.group("b2"))
                terms.append(max(_f(a), _f(b))); flags.append("asymmetric")
            pos = mt.end()
        if not terms:
            continue
        start = mv.start()
        if text[max(0, start - 1):start] == "(" and text[pos:pos + 1] == ")":   # (1.2 ± 0.3) × 10^-5
            start -= 1; pos += 1
        value, sigma = _f(mv.group(0)), math.sqrt(sum(t * t for t in terms))
        if len(terms) > 1:
            flags.append("quadrature")
        mci = _CI_TAG.match(text, pos)
        if mci:
            sigma /= 1.959963984540054; flags.append("ci_halfwidth"); pos = mci.end()
        mp = _POW_RE.match(text, pos)
        if mp:
            f = 10.0 ** _f(mp.group("e")); value *= f; sigma *= f; pos = mp.end(); flags.append("power_of_ten")
        if "e" in mv.group(0) or "E" in mv.group(0):
            flags.append("power_of_ten")
        unit, end = _unit_at(text, pos)
        if not free(start, end):
            continue
        rec = _emit(text, start, end, value, sigma, unit, flags)
        if rec:
            out.append(rec); taken.append((start, end))

    # 3. ranges "3.2–4.8 GeV" as a uniform distribution
    for m in _RANGE.finditer(text):
        lo, hi = _f(m.group("lo")), _f(m.group("hi"))
        if hi <= lo:
            continue
        flags = ["range_uniform"]
        pos = m.end()
        mp = _POW_RE.match(text, pos)
        f = 1.0
        if mp:
            f = 10.0 ** _f(mp.group("e")); pos = mp.end(); flags.append("power_of_ten")
        unit, end = _unit_at(text, pos)
        if unit is None or not free(m.start(), end):
            continue
        rec = _emit(text, m.start(), end, (lo + hi) / 2 * f, (hi - lo) * f / (2 * math.sqrt(3)), unit, flags)
        if rec:
            out.append(rec); taken.append((m.start(), end))

    out.sort(key=lambda r: r["span"])
    return out
