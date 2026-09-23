"""fleet_oed.next_best_acquisition(method="batch"): hoisted G/σ_min(G) and a stacked eigh give the loop's ranking bit for bit."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_engine.fleet_oed import next_best_acquisition  # noqa: E402


@pytest.mark.parametrize("seed", range(30))
def test_batch_ranking_equals_loop(seed):
    rng = np.random.default_rng(seed)
    nm = int(rng.integers(1, 30)); nc = int(rng.integers(0, 30))
    M = np.abs(rng.standard_normal((nm, int(rng.integers(1, 40))))) * (rng.random((nm, 1)) < 0.7)
    C = [np.abs(rng.standard_normal(nm)) * (rng.random(nm) < 0.3) for _ in range(nc)]
    if nc > 2:
        C[1] = C[0].copy()                                  # exact tie: stable order must survive
    names = None if seed % 2 else [f"n{i}" for i in range(nc)]
    assert repr(next_best_acquisition(M, C, names)) == repr(next_best_acquisition(M, C, names, method="batch"))


def test_degenerate_input_fails_the_same_way():
    M = np.eye(3); C = [np.array([np.nan, 0, 0]), np.ones(3)]
    for m in ("direct", "batch"):
        with pytest.raises(ValueError):
            next_best_acquisition(M, C, method=m)
    with pytest.raises(ValueError):
        next_best_acquisition(M, [np.ones(3)], method="fast")
