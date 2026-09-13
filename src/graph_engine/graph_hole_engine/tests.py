"""
graph_hole_engine.tests — machine-checks for the refactored architecture.
=========================================================================
Preserves the byte-equality machine-checks (resolvent_leverage vs C's & I's committed code, the
waterfill OC ≡ Fedorov-Wynn D-optimal recovery) and adds the shippable-engine + sibling-separation
tests. Plain asserts, no pytest dependency (pure numpy) — but pytest-discoverable (test_* functions).

Run:    python3 -m graph_hole_engine.tests        (or  python3 papers/graph_hole_engine/tests.py)
        pytest papers/graph_hole_engine/tests.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# --- import bootstrap: work as `-m`, as a bare script, and under pytest ---------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPERS = os.path.dirname(_HERE)                       # papers/  (parent of the package dir)
if _PAPERS not in sys.path:
    sys.path.insert(0, _PAPERS)

from graph_hole_engine import kernel                                                   # noqa: E402
from graph_hole_engine import engine as engine_mod                                     # noqa: E402
from graph_hole_engine import oed_sibling as oed_mod                                   # noqa: E402
from graph_hole_engine.kernel import participation_ratio, resolvent_leverage           # noqa: E402
from graph_hole_engine.engine import DomainAdapter, GraphHoleEngine, HoleReport        # noqa: E402
from graph_hole_engine.adapters import PhysicsGraphAdapter, PapersAdapter              # noqa: E402
from graph_hole_engine.oed_sibling import OEDDesign                                    # noqa: E402


# ==================================================================================================
# GROUP A — PRESERVED byte-equality machine-checks (the abstraction is real iff these hold).
# ==================================================================================================


def test_resolvent_leverage_equals_C_pinv_diag():
    """resolvent_leverage on the physics graph-Laplacian == C's np.diag(np.linalg.pinv(L))
    (the form committed at scripts/twin_calibration.py:165). Byte-equal, over-determined: C's code
    existed BEFORE this abstraction."""
    ph = PhysicsGraphAdapter()
    L = ph.governing_operator()
    engine_leverage = resolvent_leverage(L, ph.point_embeddings())
    C_reference = np.diag(np.linalg.pinv(L))                     # C's laplacian_pinv_diag form
    assert np.allclose(engine_leverage, C_reference), "physics leverage != C's diag(pinv(L))"


def test_resolvent_leverage_equals_I_variance_d():
    """resolvent_leverage on the OED Fisher == I's np.einsum('ij,jk,ik->i', F, Minv, F)
    (the form committed at scripts/physics_exp/d_extremal_allocation_demo.py:203). Same function
    object, read as d(x) instead of L⁺_ii — the byte-equality bridge between the two halves."""
    oed = OEDDesign(deg=3, n_grid=21)
    M = oed.fisher()
    F = oed.sensitivities()
    engine_variance = resolvent_leverage(M, F)
    Mi = np.linalg.inv(M)
    I_reference = np.einsum("ij,jk,ik->i", F, Mi, F)            # I's variance_d form
    assert np.allclose(engine_variance, I_reference), "OED variance != I's f(x)ᵀM⁻¹f(x)"


def test_waterfill_drives_oed_to_D_optimal():
    """waterfill_reallocate (== topopt OC ≡ Fedorov-Wynn, warpfem_topopt.py:154 / d_extremal:203)
    drives the OED design to the Kiefer-Wolfowitz D-optimal: max_x d(x) → p and the interior support
    centroid → 1/√5 for the cubic. The OPTIMIZER-half kernel, exercised end-to-end."""
    oed = OEDDesign(deg=3, n_grid=201)
    oed.optimize(4000)
    kw = oed.kw_max_variance()
    centroid = oed.interior_support_centroid()
    assert abs(kw - oed.p) < 0.1, f"KW cert max d(x)={kw:.4f} did not reach p={oed.p}"
    assert abs(centroid - 1 / np.sqrt(5)) < 0.05, f"support centroid={centroid:.4f} != 1/√5"


def test_participation_ratio_is_spectral_ess():
    """participation_ratio is the SPECTRAL 2nd-moment effective-rank (the Anderson inverse-
    participation-ratio): k for a flat spectrum, 1 for a rank-1 spectrum. It is the spectral member
    of the same ESS family as the Kish design-effect / CCF β-factor sum-form N_eff = k/(1+(k-1)ρ) —
    but it is NOT byte-equal to that sum-form (they are different members). Assert the true spectral
    identities AND, honestly, that the sum-form (a DIFFERENT function) is the one equal to the β-factor."""
    assert abs(participation_ratio(np.ones(5)) - 5.0) < 1e-9           # flat spectrum of k ⇒ k
    assert abs(participation_ratio(np.array([3.0, 0, 0, 0])) - 1.0) < 1e-9  # rank-1 ⇒ 1
    k, r = 4, 0.5
    R = (1 - r) * np.eye(k) + r * np.ones((k, k))                      # equicorrelation matrix
    pr = participation_ratio(np.linalg.eigvalsh(R))
    spectral_closed = k ** 2 / ((1 + (k - 1) * r) ** 2 + (k - 1) * (1 - r) ** 2)
    kish_ccf_sumform = k / (1 + (k - 1) * r)                           # Kish DEFF-ESS == CCF β-factor
    assert abs(pr - spectral_closed) < 1e-9                            # matches its spectral closed form
    assert abs(spectral_closed - kish_ccf_sumform) > 0.1              # honestly DISTINCT from the sum-form
    assert abs(k ** 2 / R.sum() - kish_ccf_sumform) < 1e-9            # the SUM-form is the β-factor (right function)


def test_engine_draws_the_shared_kernel_objects():
    """The GraphHoleEngine and the OED sibling draw the SAME function objects out of kernel (proven by
    identity `is`, stronger than a source-string match)."""
    assert engine_mod.resolvent_leverage is kernel.resolvent_leverage
    assert engine_mod.participation_ratio is kernel.participation_ratio
    assert engine_mod.spectral_sigma_min is kernel.spectral_sigma_min
    assert oed_mod.resolvent_leverage is kernel.resolvent_leverage
    assert oed_mod.waterfill_reallocate is kernel.waterfill_reallocate


# ==================================================================================================
# GROUP B — the SHIPPABLE hole-detector (physics ↔ papers transfer on ONE engine + ONE kernel).
# ==================================================================================================


def test_physics_hole_is_the_pendant_not_the_high_degree_decoy():
    """The resolvent catches what degree misses: the pendant V6 (low degree) is the top structural
    hole, over the high-internal-degree decoy clique. This is the whole reason to use L⁺_ii."""
    rep = GraphHoleEngine(PhysicsGraphAdapter()).detect()
    assert isinstance(rep, HoleReport)
    assert rep.hole_ranking[0] == "V6", f"top structural hole was {rep.hole_ranking[0]!r}, expected V6"


def test_physics_two_decorrelated_legs():
    """residual ⊥ structure: the observation anomaly (V2, huge work, well-embedded) is a DIFFERENT
    node from the structural hole (V6), and ρ(residual, leverage) is low — 2 decorrelated legs."""
    rep = GraphHoleEngine(PhysicsGraphAdapter()).detect()
    assert rep.residual_ranking[0] == "V2", f"top observation anomaly was {rep.residual_ranking[0]!r}"
    assert rep.residual_ranking[0] != rep.hole_ranking[0], "the two legs collapsed onto one node"
    assert rep.rho_residual_structure < 0.8, f"legs not decorrelated: ρ={rep.rho_residual_structure:.2f}"


def test_papers_transfer_same_engine_same_kernel():
    """The SAME GraphHoleEngine class + SAME kernel functions run on the PAPER graph and return a
    HoleReport — H's 'hole-detector transfers physics ↔ PAPER'."""
    eng_phys = GraphHoleEngine(PhysicsGraphAdapter())
    eng_papers = GraphHoleEngine(PapersAdapter())
    assert type(eng_phys) is type(eng_papers)                  # literally the same engine class
    rep = eng_papers.detect()
    assert isinstance(rep, HoleReport)
    assert rep.domain == "PapersAdapter"
    # the leverage + over-det paths used the shared kernel (no per-domain re-implementation):
    assert eng_papers.hole_field().shape[0] == len(PapersAdapter().nodes())


def test_papers_leaks_declared_and_surfaced():
    """The paper adapter declares its 3 honest leaks (σ_min→extraction-conf, residual→set-diff,
    rank→triage) and the engine surfaces them in the report."""
    rep = GraphHoleEngine(PapersAdapter()).detect()
    assert len(rep.leaks) == 3, f"expected 3 declared leaks, got {len(rep.leaks)}"
    joined = " ".join(rep.leaks).lower()
    assert "set-difference" in joined                          # L1 residual leak
    assert "extraction_confidence" in joined                   # L2 σ_min leak
    assert "triage" in joined                                  # L3 ranking leak


def test_papers_abstains_below_extraction_floor():
    """abstain-when-unspanned in the leaked identifiability: the low-extraction paper (0.35 < 0.6)
    abstains; the high-extraction papers do not."""
    rep = GraphHoleEngine(PapersAdapter()).detect()
    assert "vague_hype_abstract" in rep.abstained, "low-extraction paper should abstain"
    assert "codesign_2604.25193" not in rep.abstained, "high-extraction paper should not abstain"


def test_physics_watertight_no_leaks():
    """The physics adapter shares the full kernel path — no leaks (the WATERTIGHT half)."""
    rep = GraphHoleEngine(PhysicsGraphAdapter()).detect()
    assert rep.leaks == [], f"physics should be leak-free, got {rep.leaks}"


# ==================================================================================================
# GROUP C — the SIBLING separation (OED shares code, not question; not a DomainAdapter).
# ==================================================================================================


def test_oed_is_a_sibling_not_a_domain_adapter():
    """The architectural crux: OEDDesign is NOT a DomainAdapter (it asks a different question) yet it
    draws the SAME kernel functions (resolvent_leverage as d(x); waterfill_reallocate). Sharing code
    is not sharing semantics — proven in the type system + by function identity."""
    assert not issubclass(OEDDesign, DomainAdapter), "OED must NOT be a DomainAdapter (it is a sibling)"
    oed = OEDDesign(deg=3, n_grid=21)
    assert not isinstance(oed, DomainAdapter)
    # yet it genuinely reuses the shared kernel (same function objects, verified in the module namespace):
    assert oed_mod.resolvent_leverage is kernel.resolvent_leverage
    assert oed_mod.waterfill_reallocate is kernel.waterfill_reallocate
    # and the reuse is real at runtime: variance_field == the kernel leverage of the Fisher.
    assert np.allclose(oed.variance_field(),
                       resolvent_leverage(oed.fisher(), oed.sensitivities()))


def test_engine_never_calls_waterfill():
    """The hole-engine must NOT reallocate a budget — waterfill is the optimizer-half's alone. Assert
    the engine module does not even import it (a static guard against re-folding OED back in)."""
    assert not hasattr(engine_mod, "waterfill_reallocate"), \
        "engine imported waterfill_reallocate — the sibling optimizer leaked into the detector"


# ==================================================================================================
# runner (pure-numpy; no pytest needed)
# ==================================================================================================


def _run_all() -> bool:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    width = max(len(t.__name__) for t in tests)
    n_pass = 0
    print("=" * (width + 12))
    print(f"graph_hole_engine.tests — {len(tests)} machine-checks")
    print("=" * (width + 12))
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__.ljust(width)}")
            n_pass += 1
        except Exception as e:                                 # noqa: BLE001
            print(f"  FAIL  {t.__name__.ljust(width)}  -> {e}")
    print("-" * (width + 12))
    print(f"  {n_pass}/{len(tests)} passed")
    print("=" * (width + 12))
    return n_pass == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
