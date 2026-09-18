"""record_guarantee: the admitted error rate stays under α on held-out records, where the product gate does not."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "examples" / "engine_experiments"))
import pytest
pytest.importorskip("sklearn")
from graph_engine.record_guarantee import least_certain_field_score, fit_threshold, admit  # noqa: E402
from graph_engine.typed_extraction import extract  # noqa: E402
from graph_engine.lens_pooling import balanced_lenses  # noqa: E402
import e12_documents_to_next_action as e12  # noqa: E402

FIELDS = [("atom", "a"), ("atom", "b"), ("atom", "c"), ("polarity", "ax")]; BAL = balanced_lenses({"option_order": 2, "phrasing_a": 2, "phrasing_b": 2})


def _records(n, q, seed):
    rng = np.random.default_rng(seed); P, ok = [], []
    for _ in range(n):
        tr = {f: bool(rng.integers(2)) for f in FIELDS}; ex = extract("", FIELDS, e12.make_judge(rng, tr, q), BAL)
        P.append([ex["p"][f] for f in FIELDS]); ok.append(all((ex["p"][f] > 0.5) == tr[f] for f in FIELDS))
    return np.array(P), np.array(ok)


def test_admitted_error_rate_is_controlled_where_the_product_gate_is_not():
    """Measured: 4 fields, 10 % shared misreading → ~0.64 of records entirely right. The product gate at nominal 0.9 admits
    records that are right ~0.65 of the time (level broken in every split); the guarantee at α = 0.1 admits nothing, and at a
    reachable α = 0.45 holds in every split."""
    broken_prod, refused, broken_conf, cov = 0, 0, 0, []
    for seed in range(20):
        Pc, okc = _records(600, 0.1, seed); Pt, okt = _records(600, 0.1, 100 + seed); s_c, s_t = least_certain_field_score(Pc), least_certain_field_score(Pt)
        prod = np.prod(np.maximum(Pt, 1 - Pt), 1) >= 0.9; broken_prod += prod.sum() > 0 and (1 - okt[prod].mean()) > 0.1
        refused += fit_threshold(s_c, okc, alpha=0.1)["admitted_in_calibration"] == 0
        a = admit(s_t, fit_threshold(s_c, okc, alpha=0.45)); cov.append(a.mean()); broken_conf += (1 - okt[a].mean()) > 0.45
    assert broken_prod >= 15 and refused == 20 and broken_conf <= 1 and np.mean(cov) > 0.3


def test_threshold_is_monotone_in_alpha_and_refuses_when_nothing_is_safe():
    Pc, okc = _records(500, 0.1, 7); s = least_certain_field_score(Pc)
    t1, t2 = fit_threshold(s, okc, 0.42)["threshold"], fit_threshold(s, okc, 0.6)["threshold"]
    assert t2 >= t1
    assert fit_threshold(s, okc, 0.1)["admitted_in_calibration"] == 0                  # an unreachable target admits nothing
    assert fit_threshold(s, np.zeros_like(okc), 0.1)["admitted_in_calibration"] == 0
