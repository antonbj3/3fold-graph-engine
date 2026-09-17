"""typed_extraction: the confidence handed to the existing pipeline gate must be what the docstring says it is."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "examples" / "engine_experiments"))
pytest.importorskip("sklearn")
from graph_engine.typed_extraction import extract, agree, fields_of, to_pipeline_claim, fit_pooled_calibrator  # noqa: E402
from graph_engine.lens_pooling import balanced_lenses, error_correlation, logit  # noqa: E402
from graph_engine.paper_graph import pipeline as pp  # noqa: E402
import e12_documents_to_next_action as e12  # noqa: E402  (simulated judge with a known error structure)
from sklearn.linear_model import LogisticRegression  # noqa: E402

FIELDS = [("atom", "a"), ("atom", "b"), ("atom", "c"), ("polarity", "ax")]
BAL = balanced_lenses({"option_order": 2, "phrasing_a": 2, "phrasing_b": 2})


def _setup(q, seed):
    rng = np.random.default_rng(seed); L, Y = [], []
    for _ in range(400):
        tr = {("atom", "c"): bool(rng.integers(2))}; jd = e12.make_judge(rng, tr, q)
        L.append([logit(np.array([jd("", ("atom", "c"), l)]))[0] for l in BAL]); Y.append(tr[("atom", "c")])
    C = error_correlation(np.array(L), np.array(Y)); X, Yc = [], []
    for _ in range(600):
        tr = {("atom", "c"): bool(rng.integers(2))}; ex = extract("", [("atom", "c")], e12.make_judge(rng, tr, q), BAL, lens_corr=C)
        X.append(ex["log_odds"][("atom", "c")]); Yc.append(tr[("atom", "c")])
    return rng, C, fit_pooled_calibrator(np.array(X), np.array(Yc))


def _run(q, calibrated, two, n=1500, seed=0):
    rng, C, cal = _setup(q, seed); conf, right = [], []
    for _ in range(n):
        tr = {f: bool(rng.integers(2)) for f in FIELDS}
        ex = extract("", FIELDS, e12.make_judge(rng, tr, q), BAL, lens_corr=C, pooled_calibrator=cal if calibrated else None)
        if two: ex = agree(ex, extract("", FIELDS, e12.make_judge(rng, tr, q), BAL, lens_corr=C, pooled_calibrator=cal))
        conf.append(ex["extraction_confidence"]); right.append(all((ex["p"][f] > 0.5) == tr[f] for f in FIELDS))
    return np.array(conf), np.array(right)


def test_calibrated_confidence_is_the_probability_that_the_whole_claim_is_right():
    conf, right = _run(q=0.1, calibrated=True, two=False)
    assert abs(conf.mean() - right.mean()) < 0.06                          # measured: stated 0.60, entirely right 0.64 (0.9⁴ = 0.66)
    m = (conf >= 0.5) & (conf < 0.7)
    assert m.sum() > 1000 and abs(conf[m].mean() - right[m].mean()) < 0.07


def test_uncalibrated_confidence_overstates_when_the_judge_misreads():
    conf, right = _run(q=0.1, calibrated=False, two=False); hi = conf >= 0.7
    assert hi.mean() > 0.6 and conf[hi].mean() - right[hi].mean() > 0.12   # measured: states 0.81–0.92, is right 0.60–0.65


def test_a_second_independent_judge_admits_more_at_the_same_correctness():
    c1, r1 = _run(q=0.1, calibrated=True, two=False, seed=1); c2, r2 = _run(q=0.1, calibrated=True, two=True, seed=1)
    a1, a2 = c1 >= 0.8, c2 >= 0.8
    assert a2.mean() > a1.mean() + 0.2 and r2[a2].mean() > 0.9             # one misreading judge can hardly reach 0.8 on four fields


def test_output_runs_through_the_existing_grader_and_gate_unchanged():
    kg = pp._synthetic_kg(); node = kg["held_A"]; f = fields_of(node.claim); rng = np.random.default_rng(3)
    ex = extract("", f, e12.make_judge(rng, {x: True for x in f}, 0.0), BAL)
    pn = to_pipeline_claim(ex, pp.Paper(id="p1"), "external:lab-A", template=node.claim)
    assert pp.grade(pn, node).verdict in ("PLATFORM-COVERS", "PLATFORM-DOMINATES", "COVERS-COMPETITIVE")
    pn.extraction_confidence = 0.3
    assert pp.grade(pn, node).verdict == "ABSTAIN"
