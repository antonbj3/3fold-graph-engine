#!/usr/bin/env python3
"""
typed_extraction.py — the prose → structured-claim step of paper_graph/pipeline.py, with a confidence that means something.

paper_graph/pipeline.py takes a structured `Claim` plus an `extraction_confidence` and refuses to touch the knowledge
graph when that confidence is below TAU_EXTRACT = 0.6. Its register marks the extraction itself as STUBBED. This module
is that step, reduced to what can be specified and tested without committing to a particular model:

  fields     every mechanism atom and every polarity axis of the TARGET node's claim becomes one yes/no field
             ("does the text claim <atom>?", "does it assert <axis> or assert against it?").
  judge      any callable  judge(text, field, lens) -> P(yes).  A lens is one way of asking (phrasing, option order);
             lens sets come from lens_pooling.balanced_lenses so that additive nuisance biases cancel in the sum.
  pooling    per field, log-odds are summed over lenses with weight N_eff/K (lens_pooling.tempered_weight) — repeated or
             same-order lenses are not independent votes. An optional calibrator (lens_pooling.StratifiedPlatt) is
             applied before pooling.
  calibrate  `pooled_calibrator` (from `fit_pooled_calibrator`, labelled fields) maps the pooled log-odds to calibrated ones.
             It is what puts the judge's MISREADING rate into p: a judge that misreads 10 % of fields can never be more
             than 90 % sure of one, however many lenses agree. The map is ISOTONIC, not a straight line in log-odds: the
             true relation saturates, and a linear (Platt) fit was tried first and came out under-confident — it stated
             0.44 for four-field claims that were entirely right 0.63 of the time (0.9⁴ = 0.66). The test caught it.
  confidence extraction_confidence = Π_f max(p_f, 1 − p_f) over the fields of the target node.
             If the p_f are calibrated and field errors are independent this IS P(every field of the extracted claim is
             right); if field errors are positively associated it is a lower bound. So the pipeline's existing gate
             "confidence ≥ 0.6" reads: at least a 60 % chance that the claim handed to the grader is entirely right.

`extract` returns a plain dict; `to_pipeline_claim` turns it into pipeline.Claim / PaperNode so the existing grader, ABSTAIN
gate and C_pred update run unchanged. What is NOT here: a judge. examples/engine_experiments/e12 uses a simulated one with
known error structure; e7 measured a local 0.5B model on the lens properties this module relies on.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .lens_pooling import error_correlation, logit, tempered_weight

__all__ = ["fields_of", "extract", "agree", "fit_pooled_calibrator", "to_pipeline_claim"]


def fields_of(target_claim: Any) -> list[tuple[str, str]]:
    """[("atom", name), ..., ("polarity", axis), ...] for a pipeline.Claim (or any object / dict with those attributes)."""
    get = (lambda k: target_claim.get(k, [])) if isinstance(target_claim, dict) else (lambda k: getattr(target_claim, k, []))
    return [("atom", a) for a in get("mechanism")] + [("polarity", ax) for ax in dict(get("polarity") or {})]


def extract(text: str, fields: Sequence[tuple[str, str]], judge: Callable[[str, tuple[str, str], Any], float],
            lenses: Sequence[Any], lens_corr: np.ndarray | None = None, calibrate: Callable[[np.ndarray, Sequence], np.ndarray] | None = None,
            pooled_calibrator: Callable[[np.ndarray], np.ndarray] | None = None) -> dict[str, Any]:
    """P[f, l] = judge(text, field f, lens l). `lens_corr` is the K×K error-correlation of the lenses (from a labelled
    calibration set, lens_pooling.error_correlation); without it lenses are pooled as if independent."""
    P = np.array([[judge(text, f, l) for l in lenses] for f in fields], float)
    L = calibrate(P, fields) if calibrate is not None else logit(P)
    w = tempered_weight(lens_corr) if lens_corr is not None else 1.0
    pooled = w * L.sum(1)
    if pooled_calibrator is not None:                      # pooled log-odds → calibrated log-odds (fit on labelled fields)
        pooled = np.asarray(pooled_calibrator(pooled), float)
    p = 1.0 / (1.0 + np.exp(-pooled))
    conf = float(np.prod(np.maximum(p, 1 - p))) if len(p) else 0.0
    return {"p": dict(zip(fields, p.tolist())), "log_odds": dict(zip(fields, pooled.tolist())), "extraction_confidence": conf, "lens_weight": float(w),
            "mechanism": [k for (kind, k), q in zip(fields, p) if kind == "atom" and q > 0.5],
            "polarity": {k: (1 if q > 0.5 else -1) for (kind, k), q in zip(fields, p) if kind == "polarity"}}


def fit_pooled_calibrator(pooled_log_odds: np.ndarray, truth: np.ndarray, floor: float = 1e-3) -> Callable[[np.ndarray], np.ndarray]:
    """Monotone map pooled log-odds → calibrated log-odds, fitted on labelled fields (truth = the field's true yes/no)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=floor, y_max=1 - floor, out_of_bounds="clip").fit(np.asarray(pooled_log_odds, float), np.asarray(truth, float))
    def f(z):
        p = np.clip(iso.predict(np.atleast_1d(np.asarray(z, float))), floor, 1 - floor)
        return np.log(p / (1 - p))
    return f


def agree(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two extractions of the same text by judges with DIFFERENT origins (different model, or a non-text channel).
    Calibrated log-odds of independent channels add; where the two disagree they cancel, the field falls to p ≈ 0.5 and
    the product confidence drops below the gate. An error shared by every lens of one judge — a misreading — cannot be
    seen from inside that judge (measured in e12: harmful graph writes 3.7 of 43 papers at a 10 % misreading rate, with
    any number of lenses). This is the admission rule of this repository applied to extraction: two decorrelated channels."""
    fields = list(a["log_odds"]); lo = np.array([a["log_odds"][f] + b["log_odds"][f] for f in fields]); p = 1 / (1 + np.exp(-lo))
    return {"p": dict(zip(fields, p.tolist())), "log_odds": dict(zip(fields, lo.tolist())),
            "extraction_confidence": float(np.prod(np.maximum(p, 1 - p))) if len(p) else 0.0, "lens_weight": float("nan"),
            "mechanism": [k for (kind, k), q in zip(fields, p) if kind == "atom" and q > 0.5],
            "polarity": {k: (1 if q > 0.5 else -1) for (kind, k), q in zip(fields, p) if kind == "polarity"}}


def to_pipeline_claim(extracted: dict[str, Any], paper: Any, lineage: str, template: Any = None):
    """(pipeline.PaperNode) — scope / grounding / axes are copied from `template` (the target node's claim) for the atoms
    that were extracted; they are properties of the atom vocabulary, not of the paper's prose."""
    from .paper_graph import pipeline as pp
    t = template
    keep = set(extracted["mechanism"])
    claim = pp.Claim(domain=getattr(t, "domain", ""), domains=list(getattr(t, "domains", [])), mechanism=list(extracted["mechanism"]),
                     scope={k: v for k, v in getattr(t, "scope", {}).items() if k in keep},
                     atom_axis={k: v for k, v in getattr(t, "atom_axis", {}).items() if k in keep},
                     polarity=dict(extracted["polarity"]), lineage=lineage, provenance=getattr(paper, "id", ""))
    return pp.PaperNode(paper=paper, claim=claim, extraction_confidence=extracted["extraction_confidence"])
