"""
fleet_oed.py — the ACTIVE AGENT POOL OED: VIBRATE the federation's collective coverage, read σ_min(G_fleet), and output every lane's next acquisition. The one instrument that makes the
whole agent pool resonate — it composes the session's spine: cert_decorrelation (the catch-matrix / coverage), dither_schedule (the perturbation), G_fleet σ_min (the objective, D's
grundseed), and the FDT active-reader (read the soft mode by shaking, not by introspection —, the RELENTLESS σ_min-via-fluctuation program).

The agent pool is a coupled sensor bundle. Each lane contributes cert-SIGNATURES = coverage vectors over a shared basis of decorrelated FAILURE-MODES / KINDS (topology, singularity,
identifiability, recon-null, flutter, metamer, location/fluctuation/shape/dependency, …). Stack them: M (modes × lane-certs). The agent pool EVIDENCE GRAMIAN G = M Mᵀ; σ_min(G) = the
LEAST-covered direction = the pool's SOFT MODE = its biggest collective blind spot (confidence-blind). ★VIBRATE: perturb the coverage and read the
σ_min response — the direction that fluctuates most under perturbation IS the soft mode (FDT, model-free). ★ACQUIRE (E-optimal OED): the candidate acquisition (a new cert/dataset
= a coverage column) that maximally LIFTS σ_min(G) is the pool's next-best-action; rank candidates by Δσ_min. This turns D's G_fleet objective + A's decorrelated-coverage policy
into a runnable dispatch: read the blind kind, acquire it, repeat — the agent pool resonating toward full coverage.
"""
from __future__ import annotations
import numpy as np
from .linalg_guards import guarded_eigh, guarded_eigvalsh

__all__ = ["coverage_gramian", "fleet_soft_mode", "delta_sigmin", "next_best_acquisition", "vibrate", "fleet_oed_report"]


def coverage_gramian(M):
    """G_fleet = M Mᵀ over the failure-mode basis. M is (n_modes × n_certs); each column = a cert's coverage vector (nonneg weights over the modes it catches)."""
    M = np.asarray(M, float)
    return M @ M.T


def _sigmin_eig(G):
    w, V = guarded_eigh((G + G.T) / 2); w = np.clip(w, 0, None)
    return float(w[0]), V[:, 0]


def fleet_soft_mode(M, mode_names=None):
    """the pool's SOFT MODE: σ_min(G_fleet) + the eigenvector (the least-covered direction) + its dominant failure-modes (what the whole agent pool is blind to)."""
    G = coverage_gramian(M); smin, v = _sigmin_eig(G)
    n = M.shape[0]; names = mode_names or [f"mode{i}" for i in range(n)]
    order = np.argsort(-np.abs(v))
    dom = [(names[i], round(float(v[i]), 3)) for i in order[:3]]
    smax = float(guarded_eigvalsh(G)[-1])
    return dict(sigmin=round(smin, 5), cond=round(smax / (smin + 1e-18), 1), soft_dir=v, blind_modes=dom,
                covered=(smin > 1e-9))


def delta_sigmin(M, cand):
    """rank-1 coverage update: adding candidate cert `cand` (a coverage column over modes) → G + candᵀ⊗cand → new σ_min. Returns (new_sigmin, delta)."""
    G = coverage_gramian(M); s0, _ = _sigmin_eig(G)
    c = np.asarray(cand, float).reshape(-1)
    s1, _ = _sigmin_eig(G + np.outer(c, c))
    return float(s1), float(s1 - s0)


def next_best_acquisition(M, candidates, cand_names=None):
    """E-optimal agent pool OED: rank candidate acquisitions by Δσ_min(G_fleet). The max-Δσ_min candidate is the pool's next-best-action (the acquisition that most lifts the
    least-covered direction). `candidates` = list of coverage columns (each nonneg over the modes it would newly cover)."""
    names = cand_names or [f"cand{i}" for i in range(len(candidates))]
    scored = []
    for nm, c in zip(names, candidates):
        s1, d = delta_sigmin(M, c); scored.append(dict(name=nm, new_sigmin=round(s1, 5), delta_sigmin=round(d, 5)))
    scored.sort(key=lambda x: -x["delta_sigmin"])
    return scored


def vibrate(M, n_dither=200, amp=0.1, seed=0):
    """ACTIVE (FDT) read of the soft mode WITHOUT eigendecomposition: perturb the coverage by small dithers and read which mode-direction's σ_min-response FLUCTUATES most =
    the soft mode (model-free). Returns per-mode susceptibility (response variance) — the pool's fluctuation spectrum; its peak = the blind kind. Ties the σ_min-via-fluctuation
    spine: the soft mode is the direction that MOVES most under perturbation, exactly."""
    rng = np.random.default_rng(seed); G = coverage_gramian(M); n = G.shape[0]
    s0, _ = _sigmin_eig(G)
    resp = np.zeros(n)
    for _ in range(n_dither):
        d = rng.standard_normal(n); d /= np.linalg.norm(d) + 1e-12
        s1, _ = _sigmin_eig(G + amp * np.outer(d, d))          # perturb coverage along a random mode-direction
        resp += (s1 - s0) * d * d                              # attribute the σ_min lift to the perturbed directions
    resp /= n_dither
    return resp                                                # larger where the fleet is MORE blind (perturbation there lifts σ_min most)


def fleet_oed_report(M, candidates=None, mode_names=None, cand_names=None):
    """the runnable agent pool dispatch: {soft_mode (blind kind), σ_min(G_fleet), next-best acquisition ranking}. Every lane reads this to know where the agent pool is blind + what to build."""
    sm = fleet_soft_mode(M, mode_names)
    out = dict(sigmin=sm["sigmin"], cond=sm["cond"], blind_modes=sm["blind_modes"], covered=sm["covered"])
    if candidates is not None:
        out["acquisition_ranking"] = next_best_acquisition(M, candidates, cand_names)
        out["fleet_nba"] = out["acquisition_ranking"][0]["name"] if out["acquisition_ranking"] else None
    return out


def _selftest():
    ok = True
    modes = ["topology", "singularity", "identifiability", "recon-null", "flutter", "metamer", "shape", "dependency"]
    n = len(modes)
    # a agent pool that covers everything EXCEPT 'dependency' (mode 7) → soft mode should point at dependency, NBA = the candidate covering it
    def onehot(i, w=1.0): c = np.zeros(n); c[i] = w; return c
    M = np.column_stack([onehot(i) for i in range(n - 1)] + [onehot(0, 0.5)])   # modes 0..6 covered, mode 7 (dependency) UNCOVERED
    sm = fleet_soft_mode(M, modes)
    c1 = (sm["sigmin"] < 1e-9) and (sm["blind_modes"][0][0] == "dependency")
    ok &= c1; print(f"  soft mode: σ_min={sm['sigmin']} blind={sm['blind_modes'][0]} (expect dependency, σ_min≈0) {'OK' if c1 else 'FAIL'}")

    # candidate acquisitions: one covers 'dependency' (should win), one covers already-covered 'topology' (Δ≈0, A's same-kind=+0)
    cands = [onehot(7), onehot(0), onehot(6)]; cnames = ["acquire-dependency", "acquire-topology(redundant)", "acquire-shape(redundant)"]
    rank = next_best_acquisition(M, cands, cnames)
    c2 = rank[0]["name"] == "acquire-dependency" and rank[0]["delta_sigmin"] > 0 and rank[1]["delta_sigmin"] < 1e-6
    ok &= c2; print(f"  NBA ranking: top={rank[0]['name']} (Δσ_min={rank[0]['delta_sigmin']}), redundant Δ={rank[1]['delta_sigmin']} {'OK' if c2 else 'FAIL'}")

    # vibrate (FDT): the fluctuation spectrum peaks at the blind mode (dependency, index 7) WITHOUT eigendecomposition
    resp = vibrate(M); c3 = int(np.argmax(resp)) == 7
    ok &= c3; print(f"  vibrate (model-free): peak mode={modes[int(np.argmax(resp))]} (expect dependency) {'OK' if c3 else 'FAIL'}")

    # after acquiring dependency, the agent pool is fully covered (σ_min lifts off 0)
    M2 = np.column_stack([M, onehot(7)]); sm2 = fleet_soft_mode(M2, modes)
    c4 = sm2["covered"] and sm2["sigmin"] > sm["sigmin"]
    ok &= c4; print(f"  after acquisition: σ_min {sm['sigmin']}→{sm2['sigmin']} covered={sm2['covered']} {'OK' if c4 else 'FAIL'}")

    # full report
    rep = fleet_oed_report(M, cands, modes, cnames)
    c5 = rep["fleet_nba"] == "acquire-dependency" and not rep["covered"]
    ok &= c5; print(f"  report: fleet_nba={rep['fleet_nba']} covered={rep['covered']} {'OK' if c5 else 'FAIL'}")

    print(f"\n  fleet_oed self-test: {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _selftest()
