"""plan_value: settle cost as a dynamic programme, checked against closed forms; the cheap instrument is worth ≥ 0 and is
chosen exactly when it can settle the node often enough."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "graph_engine" / "tools")); sys.path.insert(0, str(ROOT / "tests"))
from graph_engine.plan_value import settle_cost, cheap_instrument_value, plan, expected_cost  # noqa: E402
from test_unlock_value import _graph  # noqa: E402


def test_settle_cost_closed_forms():
    assert settle_cost(0.5, [(20.0, 1.0, "cell")]) == (20.0, "cell")
    # one cheap test (cost 1, r) then the cell if still inside the band: E = 1 + P(unsettled)·20
    p, r, tau = 0.5, 0.95, 0.9
    q = p * r + (1 - p) * (1 - r); py, pn = p * r / q, p * (1 - r) / (1 - q)
    unsettled = (q if (1 - tau) < py < tau else 0) + ((1 - q) if (1 - tau) < pn < tau else 0)
    s, first = settle_cost(p, [(1.0, r, "judge"), (20.0, 1.0, "cell")], tau)
    assert first == "judge" and abs(s - (1 + unsettled * 20)) < 1e-9
    assert cheap_instrument_value(0.5, (1.0, 0.95), (20.0, 1.0)) > 15


def test_cheap_instrument_is_never_harmful_and_is_skipped_when_useless():
    for p in np.linspace(0.11, 0.89, 15):
        for r in (0.6, 0.75, 0.9, 0.99):
            assert cheap_instrument_value(p, (1.0, r), (20.0, 1.0)) >= -1e-12
    assert settle_cost(0.5, [(1.0, 0.6, "weak"), (2.0, 1.0, "cell")])[1] == "cell"          # a weak judge cannot leave the band: not used
    assert settle_cost(0.5, [(30.0, 0.95, "dear"), (20.0, 1.0, "cell")])[1] == "cell"       # dearer than the cell: not used


def test_plan_uses_settle_costs_and_names_the_first_instrument():
    g = _graph([0.5, 0.85, 0.85], [1, 1, 1]); P = {"n0": 0.5, "n1": 0.85, "n2": 0.85}      # 0.85 < tau: still to settle
    ins = {i: [(1.0, 0.95, "judge"), (20.0, 1.0, "cell")] for i in P}
    pl = plan(g, P, ins)
    assert pl["first_node"] == "n0" and pl["first_instrument"] == "judge"
    only_cell = {i: [(20.0, 1.0, "cell")] for i in P}
    assert expected_cost(g, P, ins) < expected_cost(g, P, only_cell)
