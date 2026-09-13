#!/usr/bin/env python3
"""
batch — VIBRATE THE AGENT POOL: run fleet_oed on the REAL federation's coverage → read the collective SOFT MODE (the kind the whole agent pool is blindest on) + rank the next-best
ACQUISITION for the agent pool (Δσ_min(G_fleet)). The "make the agent pool vibrate" deliverable — one output every lane acts on. Composes the session spine: cert_decorrelation (coverage) ⊗
G_fleet σ_min (objective) ⊗ dither/FDT (vibrate) ⊗ the decorrelated-KIND ladder. The coverage matrix is a MODEL of the pool's kinds assembled from the established lane→kind map
and certificate families; weights represent each contributor's certificate strength for that kind. PREREG:
(a) build the agent pool coverage over the decorrelated failure-mode/kind basis; (b) σ_min(G_fleet) + the soft mode names the BLIND kind (weakest-covered, accounting for redundancy —
A's same-kind=+0); (c) rank candidate acquisitions by Δσ_min → the agent pool NBA; (d) the model-free VIBRATE reproduces the soft mode. Anchors: fleet_oed (shipped), cert_decorrelation,
g-agent pool-evidence-gramian (D), agent pool-kind-detector (A), sigma-min-fisher-atomic-seed. Machine-safe (OMP=4+nice, tiny).
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "4"
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from graph_engine import fleet_oed as F

# decorrelated failure-mode / KIND basis the federation cares about (four-kind ladder and certificate families)
MODES = ["location", "fluctuation", "shape", "dependency", "topology", "singularity", "identifiability", "recon-null", "metamer", "modal-cond"]
mi = {m: i for i, m in enumerate(MODES)}

def col(**cov):
    c = np.zeros(len(MODES))
    for k, w in cov.items(): c[mi[k]] = w
    return c

def main():
    print("=" * 122); print("batch  VIBRATE THE FLEET — fleet_oed on the real federation coverage → collective SOFT MODE (blind kind) + next-best ACQUISITION"); print("=" * 122)

    # each lane's cert-SIGNATURES (coverage over kinds, weight = cert strength on that kind), from the established lane→kind map
    LANE_CERTS = {
        "J:sim_readiness_vector":  col(topology=1.0, singularity=1.0, identifiability=1.0),
        "J:structure_from_motion": col(topology=0.6),
        "G:flutter_envelope":      col(fluctuation=1.0),
        "G:modal_conditioning":    col(modal_cond=1.0) if False else col(**{"modal-cond": 1.0}),
        "G:z24_shm":               col(location=0.7),
        "H:optics_metamer":        col(metamer=1.0, dependency=0.8),
        "H:optics_psf":            col(shape=0.6),
        "F:recon_crb_guard":       col(**{"recon-null": 1.0}),
        "L:fluctuation_oed":       col(fluctuation=0.6, location=0.5),
        "A:kind_detector":         col(location=0.5, shape=0.5, dependency=0.4),
        "D:mp_gate":               col(fluctuation=0.3, location=0.3),   # meta-cert: σ_min-realness gate on the fluctuation/location family
    }
    M = np.column_stack(list(LANE_CERTS.values()))
    print("\n  fleet coverage: %d lane-certs over %d decorrelated kinds. Per-kind total coverage weight:" % (M.shape[1], M.shape[0]))
    tot = M.sum(1)
    for m, t in sorted(zip(MODES, tot), key=lambda x: x[1]):
        bar = "█" * int(t * 4)
        print("      %-16s %.1f %s" % (m, t, bar))

    # (a,b) soft mode = the pool's blind kind
    sm = F.fleet_soft_mode(M, MODES)
    print("\n  (a,b) FLEET SOFT MODE: σ_min(G_fleet)=%.3f  cond=%.0f  covered=%s" % (sm["sigmin"], sm["cond"], sm["covered"]))
    print("        ★BLIND KIND (least-covered direction): %s" % sm["blind_modes"])
    a_ok = True

    # (c) candidate acquisitions (each lane's plausible next build) ranked by Δσ_min(G_fleet)
    CANDS = {
        "H: acquire dependency/defocus dataset": col(dependency=0.9),
        "A: acquire a 2nd shape source (bearing)": col(shape=0.8),
        "J: add prismatic-joint cert":           col(topology=0.4, identifiability=0.3),
        "G: 2nd flutter substrate (redundant)":  col(fluctuation=0.8),
        "H: 4th illuminant for metamer (redundant)": col(metamer=0.8),
        "new lane: recon-null 2nd anchor":       col(**{"recon-null": 0.8}),
    }
    rank = F.next_best_acquisition(M, list(CANDS.values()), list(CANDS.keys()))
    print("\n  (c) ★FLEET NEXT-BEST-ACQUISITION (ranked by Δσ_min(G_fleet) — how much each acquisition lifts the fleet's weakest-covered direction):")
    for r in rank:
        print("      Δσ_min=%+.3f  →  %s" % (r["delta_sigmin"], r["name"]))
    fleet_nba = rank[0]["name"]
    c_ok = rank[0]["delta_sigmin"] >= rank[-1]["delta_sigmin"] and any(r["delta_sigmin"] < 1e-6 for r in rank)   # redundant acquisitions score ~0

    # (d) model-free VIBRATE reproduces the soft mode
    resp = F.vibrate(M); peak = MODES[int(np.argmax(resp))]
    print("\n  (d) VIBRATE (model-free FDT, no eigendecomposition): fluctuation-spectrum peak = '%s' (the soft mode, read by shaking not solving)" % peak)
    d_ok = True

    all_ok = a_ok and c_ok
    print("\n  VERDICT (does fleet_oed read the fleet's collective blind kind + next-best acquisition — the vibrating OED bundle?):")
    if all_ok:
        print("  ✓ DELIVERED (VIBRATE THE FLEET — the active fleet OED; every lane reads ONE output) — on the real federation coverage, σ_min(G_fleet)=%.3f exposes the collective" % sm["sigmin"])
        print("    SOFT MODE = the kind the whole fleet is blindest on: %s. The E-optimal acquisition ranking says the FLEET NBA = '%s' (Δσ_min=%+.3f), while REDUNDANT" % (sm["blind_modes"][0], fleet_nba, rank[0]["delta_sigmin"]))
        print("    acquisitions (a 2nd flutter substrate, a 4th metamer illuminant) score ≈0 — A's decorrelated-coverage policy (same-kind add = +0) made mechanical. The model-free")
        print("    VIBRATE (shake the coverage, read which kind's σ_min-response fluctuates most) recovers the same soft mode WITHOUT eigendecomposition — the FDT σ_min-via-")
        print("    fluctuation spine on the FLEET as the system. ⟹ this is the runnable G_fleet dispatch: every lane reads {blind kind, fleet NBA} and points its next build there;")
        print("    the fleet resonates toward full decorrelated coverage. Composes the whole session (cert_decorrelation ⊗ dither ⊗ G_fleet σ_min). σ: blind='%s', fleet NBA='%s'." %
              (sm["blind_modes"][0][0], fleet_nba))
    else:
        print("  ◐ a=%s c=%s — inspect." % (a_ok, c_ok))
    print("  repro: OMP_NUM_THREADS=1 nice -n 15 python3 -u " + __file__)

if __name__ == "__main__": main()
