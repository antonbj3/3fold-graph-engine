#!/usr/bin/env python3
"""
polarity_rules.py — a symbolic second channel for ONE field: the sign of the relation a sentence asserts between two quantities.

Why it exists. typed_extraction measured (e12) that an error shared by every phrasing of one judge cannot be seen from inside
that judge, and e7 measured that a 0.5B language model reads "Reducing X lowers Y" by its surface word ("lowers" → decreases):
accuracy 0.21–0.35 on sentence forms that need the two directions COMPOSED. Composition of directions is what a rule does by
construction (natural logic / monotonicity: MacCartney & Manning 2009; MonaLog, Hu et al. 2020). This channel has a different
origin from any language model, so agreement between the two is evidence that neither can give alone.

Rule. Inside a clause each quantity takes at most ONE direction word: with two or more direction words the assignment
with the smallest total token distance to the quantity spans is used ("Reducing X lowers the Y": reducing → X, lowers → Y);
a single direction word goes to the nearer quantity. Clauses: ";" and ", but" split them WITHIN one sentence;
the LAST clause that gives a direction for a quantity wins, so "it is not the case that Y decreases with X; it
increases" reads the correction. Composition never crosses a sentence end: a sentence asserts a sign only when it names BOTH
quantities ("X increases. Y decreases." asserts nothing), and sentences that disagree ABSTAIN. A clause naming neither quantity
carries a direction only when its subject is a pronoun ("…; it increases"); one naming a THIRD quantity ("…, but lower Z") is
ignored. "not"/"no"/"never" within four words before a direction word flips it, counted across a parenthetical ("does not, in any
way we could measure, increase Y" → down) and ABSTAINING on two negations; "no" before a nominal direction ("no increase in X")
denies a change rather than naming one. A quantity with no
direction word is taken as increasing ("Y falls with X"). "directly / inversely proportional" sets the sign outright; negated, it asserts nothing (ABSTAIN).
        sign = direction(X) · direction(Y).
Returns 0 (ABSTAIN) when no direction is found for Y, or when X or Y is not mentioned.

Measured on text by other people (QuaRTz test split, e15): answers 42 % (34/81), right 0.853 of those (29/34; Clopper–Pearson 95 %
0.689–0.951; the majority class among the answered items is 0.676). On the author's own templates 1.0.
Scope: English; one sentence or a short passage; explicit direction words and comparatives from a fixed lexicon; no coreference beyond "it" in
a following clause; no quantities expressed by numbers ("from 3 to 5"). It abstains rather than guess outside that scope.
"""
from __future__ import annotations

import re

__all__ = ["asserted_sign", "UP", "DOWN"]

UP = {"increase", "increases", "increased", "increasing", "rise", "rises", "rising", "rose", "grow", "grows", "growing", "grew", "higher",
      "raise", "raises", "raised", "raising", "climb", "climbs", "climbing", "more", "greater", "larger", "up", "gain", "gains", "boost", "boosts",
      # general English comparatives that name the direction of a quantity (the DOWN set holds their opposites); a comparative that names
      # the NEGATIVE attribute ("colder" when the quantity is "coldness") is read wrongly — a known limit
      "bigger", "faster", "stronger", "hotter", "warmer", "brighter", "heavier", "thicker", "wider", "longer", "deeper", "denser", "richer",
      "louder", "steeper", "harder", "quicker", "taller", "broader", "extra", "additional", "abundant"}
DOWN = {"decrease", "decreases", "decreased", "decreasing", "fall", "falls", "falling", "fell", "drop", "drops", "dropping", "dropped", "lower",
        "lowers", "lowered", "lowering", "reduce", "reduces", "reduced", "reducing", "less", "smaller", "down", "decline", "declines", "shrink", "shrinks",
        "fewer", "slower", "weaker", "colder", "cooler", "darker", "dimmer", "lighter", "thinner", "narrower", "shorter", "shallower", "sparser",
        "poorer", "quieter", "flatter", "softer", "lesser", "scarce", "limited", "lack", "little"}
_NEG = {"not", "no", "never", "n't"}
# a bare clause (one naming neither X nor Y) may only carry a direction when its subject is a pronoun: "…; it increases".
_PRONOUN = {"it", "they", "this", "that", "these", "those", "he", "she", "one", "both"}
_FUNCTION = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "does", "do", "did", "will", "would", "can",
             "could", "may", "might", "also", "then", "too", "still", "again", "much", "however", "so", "and", "but", "in",
             "of", "to", "by", "with", "as", "at", "on", "for", "its", "their", "there", "here", "if", "when", "while"}
# nominal direction words: "no increase in X was observed" states the ABSENCE of a change, not a direction — see _clause_words
_NOMINAL = {"increase", "decrease", "rise", "fall", "drop", "gain", "growth", "decline", "gains", "increases", "decreases", "rises", "falls", "drops"}


def _spans(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    p = re.findall(r"[a-z0-9']+", phrase.lower()); n = len(p)
    return [(k, k + n) for k in range(len(tokens) - n + 1) if tokens[k:k + n] == p]


def _find(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    return _spans(tokens, phrase)


def _negations(tok_c: list[str], r: int, window: int = 4) -> int:
    """How many negations govern the direction word at index r of the comma-carrying token list.

    The window is `window` words back, but a parenthetical — a stretch between two commas, "does not, in any way we could
    measure, increase Y" — is stepped over whole instead of being counted, so the negation before it is still seen.
    """
    negs = cnt = 0
    i = r - 1
    while i >= 0 and cnt < window:
        t = tok_c[i]
        if t == ",":
            j = i - 1
            while j >= 0 and tok_c[j] != ",":
                j -= 1
            i = (j - 1) if j >= 0 else (i - 1)                     # ", … ," → skip the parenthetical; a lone comma → step over it
            continue
        if t in _NEG:
            negs += 1
        cnt += 1
        i -= 1
    return negs


def _clause_words(clause: str, sx, sy):
    """Direction words of one clause as (word index, sign), or None when the clause must not be read at all.

    None is returned for a double negation ("does not not increase Y"), which no single sign can represent.
    A direction word directly after "no" and used as a noun ("no increase in X") is dropped: it denies a change rather
    than naming one.
    """
    tok_c = [t for t in re.findall(r"[a-z0-9']+|,", clause)]
    widx = [i for i, t in enumerate(tok_c) if t != ","]
    tok = [tok_c[i] for i in widx]
    words = []
    for k, t in enumerate(tok):
        d = 1 if t in UP else -1 if t in DOWN else 0
        if not d or any(a <= k < b for a, b in sx + sy):            # a direction word INSIDE a quantity name is part of the name
            continue
        r = widx[k]
        if r > 0 and tok_c[r - 1] == "no" and t in _NOMINAL:
            continue                                                # "no increase in X" — absence of a change, not a direction
        n = _negations(tok_c, r)
        if n >= 2:
            return None                                             # double negation: abstain rather than flip once
        words.append((k, -d if n == 1 else d))
    return words


def _sentence_sign(sent: str, x: str, y: str, m) -> int | None:
    """The sign one SENTENCE asserts, or None when it asserts nothing about the pair."""
    full = re.findall(r"[a-z0-9']+", sent)
    if not _find(full, x) or not _find(full, y):
        return None                                                 # both quantities must be named in the same sentence
    dx = dy = None
    for clause in re.split(r";|, but ", sent):
        tok = re.findall(r"[a-z0-9']+", clause)
        if m and m.group(0) in clause:
            if any(t in _NEG for t in tok[:tok.index(m.group(1))]):
                return 0                                            # "not inversely proportional" asserts no direction
            return 1 if m.group(1) == "directly" else -1
        sx, sy = _spans(tok, x), _spans(tok, y)
        words = _clause_words(clause, sx, sy)
        if words is None:
            return 0
        dist = lambda k, spans: min((0 if a <= k < b else min(abs(k - a), abs(k - (b - 1))) for a, b in spans), default=None)
        cx = cy = None
        if not sx and not sy:
            # a clause naming neither quantity carries a direction only if its subject is a pronoun ("it increases");
            # "…, but lower Z" names a THIRD quantity, so its direction belongs to neither X nor Y — ignore the clause
            rest = [t for t in tok if t not in _NEG and t not in _FUNCTION and t not in UP and t not in DOWN]
            if words and all(t in _PRONOUN for t in rest):
                cy = words[-1][1]
        elif len(words) >= 2 and sx and sy:
            # one direction word per quantity per clause: the assignment with the smallest total distance
            best = min(((dist(i, sx) + dist(j, sy), di, dj) for i, di in words for j, dj in words if i != j), key=lambda t: t[0])
            cx, cy = best[1], best[2]
        elif words:
            k, d = words[-1]
            if sx and (not sy or dist(k, sx) < dist(k, sy)):
                cx = d
            else:
                cy = d
        dx = cx if cx is not None else dx
        dy = cy if cy is not None else dy
    if dy is None:
        return None
    return (dx if dx is not None else 1) * dy


def asserted_sign(text: str, x: str, y: str) -> int:
    low = text.lower()
    m = re.search(r"\b(directly|inversely)\s+proportional\b", low)
    signs = []
    for sent in re.split(r"(?<=[.!?])\s+", low):
        s = _sentence_sign(sent, x, y, m if (m and m.group(0) in sent) else None)
        if s is not None:
            signs.append(s)
    if not signs or 0 in signs or len(set(signs)) > 1:
        return 0                                                    # nothing asserted, or sentences that disagree → ABSTAIN
    return signs[0]
