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
a single direction word goes to the nearer quantity. Clauses: ";", ", but" and sentence ends split
them; the LAST clause that gives a direction for a quantity wins, so "it is not the case that Y decreases with X; it
increases" reads the correction). "not"/"no"/"never" before a direction word in the same clause flips it. A quantity with no
direction word is taken as increasing ("Y falls with X"). "directly / inversely proportional" sets the sign outright; negated, it asserts nothing (ABSTAIN).
        sign = direction(X) · direction(Y).
Returns 0 (ABSTAIN) when no direction is found for Y, or when X or Y is not mentioned.

Scope: English; one sentence or a short passage; explicit direction words from a fixed lexicon; no coreference beyond "it" in
a following clause; no quantities expressed by numbers ("from 3 to 5"). It abstains rather than guess outside that scope.
"""
from __future__ import annotations

import re

__all__ = ["asserted_sign", "UP", "DOWN"]

UP = {"increase", "increases", "increased", "increasing", "rise", "rises", "rising", "rose", "grow", "grows", "growing", "grew", "higher",
      "raise", "raises", "raised", "raising", "climb", "climbs", "climbing", "more", "greater", "larger", "up", "gain", "gains", "boost", "boosts"}
DOWN = {"decrease", "decreases", "decreased", "decreasing", "fall", "falls", "falling", "fell", "drop", "drops", "dropping", "dropped", "lower",
        "lowers", "lowered", "lowering", "reduce", "reduces", "reduced", "reducing", "less", "smaller", "down", "decline", "declines", "shrink", "shrinks"}
_NEG = {"not", "no", "never", "n't"}


def _spans(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    p = re.findall(r"[a-z0-9']+", phrase.lower()); n = len(p)
    return [(k, k + n) for k in range(len(tokens) - n + 1) if tokens[k:k + n] == p]


def _find(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    return _spans(tokens, phrase)


def asserted_sign(text: str, x: str, y: str) -> int:
    low = text.lower()
    m = re.search(r"\b(directly|inversely)\s+proportional\b", low)
    dx = dy = None
    for clause in re.split(r";|, but |\. ", low):
        tok = re.findall(r"[a-z0-9']+", clause)
        if m and m.group(0) in clause:
            if any(t in _NEG for t in tok[:tok.index(m.group(1))]):
                return 0                                        # "not inversely proportional" asserts no direction
            s = 1 if m.group(1) == "directly" else -1
            return s if _find(re.findall(r"[a-z0-9']+", low), x) and _find(re.findall(r"[a-z0-9']+", low), y) else 0
        sx, sy = _spans(tok, x), _spans(tok, y)
        words = []
        for k, t in enumerate(tok):
            d = 1 if t in UP else -1 if t in DOWN else 0
            if d and not any(a <= k < b for a, b in sx + sy):       # a direction word INSIDE a quantity name is part of the name
                words.append((k, -d if any(u in _NEG for u in tok[max(0, k - 4):k]) else d))
        dist = lambda k, spans: min((0 if a <= k < b else min(abs(k - a), abs(k - (b - 1))) for a, b in spans), default=None)
        cx = cy = None
        if not sx and not sy:
            cy = words[-1][1] if words else None                   # a clause with a direction but no quantity: "it increases" → Y
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
    full = re.findall(r"[a-z0-9']+", low)
    if dy is None or not _find(full, x) or not _find(full, y):
        return 0
    return (dx if dx is not None else 1) * dy
