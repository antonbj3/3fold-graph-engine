#!/usr/bin/env python3
"""
d_oed_probe_selection_engine_v2_instance.py — a node-instance refresh: the node list reflects the
current evidence state and a probe batch is published from it.

WHAT THIS IS (a DATA refresh, not an engine change): the engine's model/objective/greedy/QC code
in d_oed_probe_selection_engine.py is imported UNCHANGED; only the graph INSTANCE (NODES) is
rebuilt from the CURRENT fold state. Decision documented: v2 instance lives in THIS separate file
(the v1 file stays intact as the landed 5/5-anchor artifact + its evidence JSON is a point-in-time
record; the engine module's functions read module globals, so this file patches eng.NODES/
NODE_BY_ID/CITED_BY and re-runs).

The node instance is rebuilt from the current evidence state: each node carries its evidence
tags (verified / partial / open / flagged) and the closed-versus-open status of the work items
that cite it.

CALIBRATION CONTEXT carried honestly: the replacement's retrospective
calibration was INCONCLUSIVE — the recorded outcome labels are too noisy; a clean calibration needs
per-probe realized-uncertainty-reduction, which is not logged. The PROSPECTIVE comparison in this
file (engine picks vs the actually-in-flight burst) is the better instrument and is emitted as
calibration data.

PRE-REGISTERED V2 ANCHORS (fixed before the run; no post-hoc moves):
  V2-A1 STALE-PURGE: no probe on PCB_MAP or PCB_R1_ROUTE appears in ANY scenario's published
        batch (they are DONE); if one does, the refresh failed.
  V2-A2 ρ-FLOOR (the settled-node anchor, unchanged from v1): the 5th same-instrument
        gpu_microbench repeat on the settled R2_DET has marginal gain < 5% of the median
        published-batch gain (gpu_rich) — a settled node's same-instrument repeat stays worthless.
  V2-A3 no same-instrument repeat on any node with ≥3 existing evidence items in any batch.
  V2-A4 greedy/exhaustive ratio ≥ 0.95 on a ≤12-node v2 sub-instance (cardinality + knapsack).
  V2-A5 stability (engine's own protocol): ρ ±50% and weights ×/÷2 → median Spearman ≥ 0.8 AND
        median top-batch Jaccard ≥ 0.6.
  V2-A6 §I-AGREEMENT (REPORT-ONLY, explicitly NOT forced and NOT gating the verdict): the audit's
        headline is the §I multi-component stitch; "engine agrees" is pre-defined as ≥3 probes on
        the §I enabler set {MULTI_STITCH, COUPLED_2WAY, FED_FLOOR, CORE2_ATTR, INTEG_STITCH,
        U4_TROPICAL, PER_VAR_PRECISION, KNOB_DERIVE, AUTONOMY_LOOP} in the gpu_rich batch.
        Disagreement is reported as a FINDING, not fixed.

Run: python3 e.py
CPU-only, deterministic (seeded). Evidence JSON →
scripts/physics_exp/artifacts/d_oed_engine_v2_refresh_evidence.json
Author: author,  (the §J refresh).
"""
import itertools
import json
import os
import random
from collections import defaultdict

import oed_probe_selection_engine as eng
from oed_probe_selection_engine import (E, PRIOR_BY_KIND, RHO_CAT, SCENARIOS)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts", "d_oed_engine_v2_refresh_evidence.json")
BATCH_PICKS = 12  # publish top-12 per scenario (task spec)

# ----------------------------------------------------------------------------------
# THE V2 GRAPH INSTANCE — rebuilt from the current fold state (see docstring sources).
# ev = existing evidence items (instrument + shared failure-mode tags; cross-worktree
# evidence carries its own author tag = genuine author-decorrelation).
# ----------------------------------------------------------------------------------
NODES_V2 = [
    # ================= §A FLOORS (all five + closures) =================
    dict(id="F_LANDAUER", kind="measured_cell", cites=[],
         desc="Landauer kTln2 tight-1% Jarzynski + REAL-GPU J/bit residual-ladder rung (folded)",
         ev=[E("cpu_numerical", "model:jarzynski", "author:D"),
             E("analytic", "model:jarzynski", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="F_PEBBLE", kind="measured_cell", cites=[],
         desc="Hong-Kung pebble floor (float4 89%, C529/C530, LBM 72 B/voxel, SNAP)",
         ev=[E("gpu_microbench", "gpu:local", "author:D", "model:hk"),
             E("gpu_microbench", "gpu:local", "author:D", "model:hk"),
             E("gpu_microbench", "gpu:local", "author:D"),
             E("analytic", "model:hk", "author:D")],
         probes=["gpu_second_device", "external_lit", "indep_reimpl"]),
    dict(id="F_CAPACITY", kind="measured_cell", cites=[],
         desc="Channel-capacity floor RUNG VERIFIED (Shannon-at-kT 1e-6; device dominates 6e6-2e9x)",
         ev=[E("cpu_numerical", "model:shannon", "author:D"),
             E("analytic", "model:shannon", "author:D")],
         probes=["external_lit", "gpu_microbench", "indep_reimpl"]),
    dict(id="F_FISHER", kind="measured_cell", cites=[],
         desc="Fisher/CRB floor (CRB 1.08, chi-theorem, N_eff=3 fleet)",
         ev=[E("cpu_numerical", "model:fisher", "author:D"),
             E("analytic", "model:fisher", "author:D"),
             E("cpu_numerical", "model:fisher", "author:D")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="F_TUR", kind="measured_cell", cites=["F_LANDAUER"],
         desc="TUR never-violated, tight on unicyclic ring (Q->1.006)",
         ev=[E("cpu_numerical", "model:langevin", "author:D"),
             E("cpu_numerical", "model:langevin", "author:D")],
         probes=["external_lit", "analytic", "indep_reimpl"]),
    dict(id="F_GEOM_RES", kind="measured_cell", cites=["F_FISHER", "R3_IDENT"],
         desc="5th floor: geometric resolution h*=min(rep-Nyquist, identifiability) VERIFIED 7/7 "
              "+ replacement re-run 355/355 identical (same instrument+model — correlated repeat)",
         ev=[E("cpu_numerical", "model:geomres", "author:D"),
             E("cpu_numerical", "model:geomres", "author:D")],
         probes=["external_lit", "indep_reimpl", "gpu_microbench"]),
    dict(id="G2B_ID_GRAZE", kind="in_flight", cites=["F_GEOM_RES"],
         desc="AUDIT item: G2b id-floor knee GRAZES (0.808 vs 0.84-class); below-floor "
              "sensitivity UNCONTROLLED — needs a decorrelated construction, not a re-run",
         ev=[E("cpu_numerical", "model:geomres", "author:D")],
         probes=["indep_reimpl", "analytic", "cpu_numerical"]),
    dict(id="U6_LANDAUER_FISHER", kind="measured_cell",
         cites=["F_LANDAUER", "F_FISHER", "U5_IDENT_TRAFFIC"],
         desc="Hole B CLOSED (H): identify-ENERGY floor E*=M*·b·kTln2 (3-currency law); "
              "guarded coincidence-of-form (setting-dependent) per the audit",
         ev=[E("cpu_numerical", "model:fisher", "author:H")],
         probes=["external_lit", "indep_reimpl", "analytic"]),
    dict(id="R4_FLOOR_HOME", kind="measured_cell", cites=["R4_CONT", "F_CAPACITY"],
         desc="Hole D RESOLVED x3 (H 47x, A balls-in-bins 566x, C corrected): R4 = JOINT "
              "capacity x provenance floor — info-independent, magnitude-coupled (loss~B/lnB)",
         ev=[E("cpu_numerical", "model:contention", "author:H"),
             E("cpu_numerical", "model:contention", "author:A"),
             E("cpu_numerical", "model:contention", "author:C")],
         probes=["external_lit", "gpu_microbench"]),
    # ================= UNIFICATION POINTS =================
    dict(id="U1_MOVE_INFO", kind="theorem_shaped", cites=["F_PEBBLE", "F_CAPACITY"],
         desc="U1 movement=information (audit-qualified: loads-given-M, rung-1 unless E_bit=wire floor)",
         ev=[E("analytic", "model:hk", "author:D")],
         probes=["indep_reimpl", "external_lit", "gpu_microbench"]),
    dict(id="U2_ERASE_TRANSPORT", kind="theorem_shaped", cites=["F_LANDAUER", "F_CAPACITY"],
         desc="U2 erasure=transport@kT, scope SHARPENED by the priced reversible corner",
         ev=[E("analytic", "author:D"),
             E("cpu_numerical", "model:adiabatic", "author:D")],
         probes=["external_lit", "cpu_numerical"]),
    dict(id="U3_FISHER_MI", kind="theorem_shaped", cites=["F_FISHER", "R3_IDENT"],
         desc="U3 Fisher=MI-local; chi-threshold unifies R3/twin/sensor",
         ev=[E("analytic", "model:fisher", "author:D")],
         probes=["indep_reimpl", "external_lit", "cpu_numerical"]),
    dict(id="U4_TROPICAL", kind="theorem_shaped", cites=[],
         desc="U4 TYPED cert composition (min-plus on monotone-lattice class; census) + held on "
              "the real composed stitch run",
         ev=[E("analytic", "author:D"), E("analytic", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["cpu_numerical", "indep_reimpl"]),
    dict(id="U5_IDENT_TRAFFIC", kind="measured_cell", cites=["F_PEBBLE", "F_FISHER"],
         desc="U5 identify-cost is a traffic floor Q*=m·b/(sigma_t^2·sigma_min^2): 4/4 real FEM (H) "
              "+ D re-ran + cyclic-network 15-sig-fig law",
         ev=[E("cpu_numerical", "model:fisher", "author:H"),
             E("cpu_numerical", "model:fisher", "author:D"),
             E("cpu_numerical", "model:kingaltman", "author:D")],
         probes=["external_lit", "gpu_microbench", "indep_reimpl"]),
    dict(id="REVERSIBLE_U2", kind="measured_cell", cites=["U2_ERASE_TRANSPORT", "F_LANDAUER"],
         desc="Reversible/adiabatic corner PRICED (X=4-158x, wire term survives; Sandia claim "
              "modeled-superseded caught) — tail#6 DONE",
         ev=[E("cpu_numerical", "model:adiabatic", "author:D"),
             E("external_lit")],
         probes=["analytic", "indep_reimpl"]),
    # ================= §B CERT-VECTOR FACETS =================
    dict(id="R1_TRAFFIC", kind="measured_cell", cites=["F_PEBBLE"],
         desc="R1 traffic cert on real kernels (float4 89% certified)",
         ev=[E("gpu_microbench", "gpu:local", "author:D", "model:hk"),
             E("gpu_microbench", "gpu:local", "author:D", "model:hk")],
         probes=["gpu_second_device", "indep_reimpl", "external_lit"]),
    dict(id="R2_DET", kind="measured_cell", cites=[],
         desc="R2 determinism SETTLED (4-branch) + H's EM/twin transfer (two-layer: operator "
              "normality (x) readout-cancellation kappa_c)",
         ev=[E("gpu_microbench", "gpu:local", "author:D", "model:detmode"),
             E("gpu_microbench", "gpu:local", "author:D", "model:detmode"),
             E("gpu_microbench", "gpu:local", "author:D", "model:detmode"),
             E("gpu_microbench", "gpu:local", "author:D", "model:detmode"),
             E("cpu_numerical", "model:detmode", "author:H")],
         probes=["gpu_microbench", "indep_reimpl", "external_lit", "cpu_numerical"]),
    dict(id="R2_NN_BOUNDED", kind="measured_cell", cites=["R2_DET", "POROELASTIC"],
         desc="THIRD roundoff class: non-normal-BOUNDED (env 77.6 >> rho, 635xeps32 "
              "non-compounding) — single instrument so far",
         ev=[E("cpu_numerical", "model:biot", "author:D")],
         probes=["gpu_microbench", "indep_reimpl", "external_lit"]),
    dict(id="R3_IDENT", kind="measured_cell", cites=["F_FISHER"],
         desc="R3 QoI identifiability + abstain-on-null-space",
         ev=[E("cpu_numerical", "model:fisher", "author:D"),
             E("analytic", "model:fisher", "author:D")],
         probes=["gpu_microbench", "external_lit"]),
    dict(id="R4_CONT", kind="measured_cell", cites=[],
         desc="R4 input-keyed REAL leg CLOSED (P1-P4 on silicon; 2.69x false-cert caught; "
              "provenance gate live; non-monotonic max_bin_freq rule 7/7)",
         ev=[E("analytic", "model:contention", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["cpu_numerical", "indep_reimpl", "gpu_second_device"]),
    dict(id="R5_GEOMETRY", kind="measured_cell", cites=["R3_IDENT", "F_GEOM_RES"],
         desc="R5 geometry facet NEW (both-direction dissociation 4/4; capsule class caught); "
              "⚠ I5 fool-instance: out-of-span QoI blind spot = proxy-completeness open",
         ev=[E("cpu_numerical", "model:geomcert", "author:D")],
         probes=["cpu_numerical", "indep_reimpl", "external_lit"]),
    dict(id="ROOFLINE", kind="measured_cell", cites=[],
         desc="roofline facet on this silicon",
         ev=[E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_second_device", "external_lit"]),
    dict(id="JBIT_ENERGY", kind="measured_cell", cites=["F_LANDAUER"],
         desc="J/bit ENERGY = independent cert axis (0.18 axis-ratio vs roofline) — single GPU run",
         ev=[E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_second_device", "external_lit", "cpu_numerical"]),
    dict(id="PER_VAR_PRECISION", kind="measured_cell", cites=["R2_DET"],
         desc="per-variable precision ACCUMULATION prefactor C(T)~sqrt(T) (measured C(5000)=824eps) "
              "CONVERGES with H's eps·sqrtT·kappa — two decorrelated instruments",
         ev=[E("gpu_microbench", "gpu:local", "author:D"),
             E("analytic", "author:H")],
         probes=["cpu_numerical", "indep_reimpl"]),
    # ================= §C CORES + LOOP MACHINERY =================
    dict(id="AUTONOMY_LOOP", kind="measured_cell",
         cites=["R1_TRAFFIC", "R2_DET", "R3_IDENT", "R4_CONT", "DEPLOY_GATE", "CORE2_ATTR"],
         desc="autonomy loop end-to-end on real nvcc CUDA (89.3% certified min-of-10)",
         ev=[E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_microbench", "indep_reimpl", "gpu_second_device"]),
    dict(id="DEPLOY_GATE", kind="measured_cell",
         cites=["R1_TRAFFIC", "R2_DET", "R3_IDENT", "R4_CONT"],
         desc="deployment-gate static SHIP/DON'T-SHIP on real solver source",
         ev=[E("cpu_numerical", "model:static_analysis", "author:D")],
         probes=["gpu_microbench", "indep_reimpl"]),
    dict(id="CORE2_ATTR", kind="measured_cell",
         cites=["R1_TRAFFIC", "R2_DET", "R3_IDENT", "R4_CONT"],
         desc="2c-real CLOSED: mechanical static attribution Jacobian, 30/30, kill-shot to "
              "machine identity; caveats: R1 column static-only (ncu blocked), R2b margin 33x",
         ev=[E("analytic", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_microbench", "cpu_numerical", "indep_reimpl"]),
    dict(id="LBM_ATTR_GPU", kind="open", cites=["CORE2_ATTR", "LBM_SOA"],
         desc="OPEN fleet follow-on: GPU-validate 2c-real's static-only LBM knob-attributions",
         ev=[E("analytic", "author:D")],
         probes=["gpu_microbench"]),
    dict(id="OED_ENGINE", kind="measured_cell", cites=[],
         desc="OED probe-selection engine landed (5/5 anchors, D re-ran; both runs same "
              "instrument+author) — §J's organ",
         ev=[E("cpu_numerical", "model:oed", "author:D"),
             E("cpu_numerical", "model:oed", "author:D")],
         probes=["indep_reimpl", "cpu_numerical", "external_lit"]),
    dict(id="OED_CALIBRATION", kind="in_flight", cites=["OED_ENGINE"],
         desc="AUDIT item: retrospective calibration INCONCLUSIVE (noisy outcome labels); needs "
              "per-probe realized-uncertainty-reduction logging + prospective pick-vs-burst "
              "comparison (THIS file emits the first prospective datapoint)",
         ev=[E("cpu_numerical", "model:oed", "author:D")],
         probes=["cpu_numerical", "indep_reimpl"]),
    dict(id="KNOB_DERIVE", kind="measured_cell", cites=["U5_IDENT_TRAFFIC"],
         desc="goal->knobs DERIVED (joint coupled O(1), never searched) executable; H "
              "conflict-method QC'd the coupling absorption",
         ev=[E("cpu_numerical", "author:D"),
             E("cpu_numerical", "author:H")],
         probes=["gpu_microbench", "indep_reimpl"]),
    # ================= §D SOLVERS + §3d TRANSFER FAMILIES =================
    dict(id="LBM_SOA", kind="measured_cell", cites=["F_PEBBLE", "R1_TRAFFIC", "ROOFLINE"],
         desc="LBM cert CLOSED: fused SoA+AoS both at 72 B/voxel, 94-99% copy roofline; "
              "mechanism corrected (fusion axis, not layout)",
         ev=[E("gpu_microbench", "gpu:local", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D"),
             E("analytic", "model:hk", "author:D")],
         probes=["gpu_second_device", "external_lit"]),
    dict(id="CONTACT_PRIV", kind="measured_cell", cites=["INTEG_STITCH", "R4_CONT"],
         desc="contact privatization DONE-with-scope (97.1% peak at contention; sparse lever = "
              "launch latency/CUDA-graphs, still open as a build)",
         ev=[E("gpu_microbench", "gpu:local", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_microbench", "gpu_second_device"]),
    dict(id="INTEG_STITCH", kind="measured_cell", cites=["U4_TROPICAL", "R4_CONT"],
         desc="REAL multi-solver stitch MEASURED (LBM+contact scenario-cert, verdicts "
              "REJECT/ABSTAIN/CERTIFY, binding flips with regime; leg-iii refuted-as-stated -> C(T))",
         ev=[E("analytic", "model:stitch", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_microbench", "cpu_numerical"]),
    dict(id="FED_FLOOR", kind="measured_cell",
         cites=["U4_TROPICAL", "LBM_SOA", "CONTACT_PRIV"],
         desc="federation floor: A's topology split structural + magnitude bottleneck-pair-"
              "dependent (co-run interference 1.003-1.116 measured); balanced-load overlap owed",
         ev=[E("analytic", "model:envelope", "author:A"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["cpu_numerical", "gpu_microbench"]),
    dict(id="COUPLED_2WAY", kind="in_flight", cites=["FED_FLOOR", "TRANSFER_LAW"],
         desc="IN-FLIGHT (burst): coupled-physics stitch — 1-way sigma_min-transport exact 1e-10 "
              "(A); 2-way FSI fixed-point = the named open boundary",
         ev=[E("cpu_numerical", "model:coupling", "author:A")],
         probes=["cpu_numerical", "analytic"]),
    dict(id="MULTI_STITCH", kind="in_flight",
         cites=["AUTONOMY_LOOP", "INTEG_STITCH", "CORE2_ATTR", "U4_TROPICAL", "FED_FLOOR",
                "COUPLED_2WAY", "PER_VAR_PRECISION", "KNOB_DERIVE"],
         desc="★§I HEADLINE (the ~30% gap): multi-variable multi-solver end-to-end autonomous "
              "generation stitch — goal->knobs->generate->certify a multi-component artifact",
         ev=[],
         probes=["gpu_microbench", "cpu_numerical"]),
    dict(id="AVBD_ROOFLINE", kind="in_flight",
         cites=["R1_TRAFFIC", "R2_DET", "CONTACT_PRIV"],
         desc="IN-FLIGHT (burst): §D KEY GAP — AVBD-class non-fluid deformable GPU kernel at "
              "roofline, certified the way LBM is (1D recipe validated CPU-only so far)",
         ev=[E("cpu_numerical", "model:avbd", "author:D")],
         probes=["gpu_microbench", "indep_reimpl"]),
    dict(id="HYPERELASTIC", kind="in_flight", cites=["TRANSFER_LAW", "R2_DET"],
         desc="IN-FLIGHT (burst): B2-L3 fp32 = GENUINE whole-path conditioning boundary "
              "(A adjudicated from cell's own kappa/e32 data; guard-to-ABSTAIN fix delivered); "
              "fixed full re-run owed",
         ev=[E("cpu_numerical", "model:fem", "author:D"),
             E("cpu_numerical", "model:fem", "author:A")],
         probes=["cpu_numerical", "indep_reimpl"]),
    dict(id="TRANSFER_LAW", kind="measured_cell", cites=["R2_DET", "R3_IDENT"],
         desc="transfer-invariance LAW x3-overdetermined (D confusion-matrix, A's S4 BC-swap w/ "
              "5th-gauge-axis scope, poroelastic instance-swap); abscissa refinement measured",
         ev=[E("cpu_numerical", "model:fingerprint", "author:D"),
             E("cpu_numerical", "model:fingerprint", "author:A"),
             E("cpu_numerical", "model:biot", "author:D")],
         probes=["external_lit", "indep_reimpl", "cpu_numerical"]),
    dict(id="POROELASTIC", kind="measured_cell", cites=["TRANSFER_LAW"],
         desc="poroelastic cert (Terzaghi exact + INDEPENDENT energy anchor; c_v-degeneracy "
              "characterized+lifted; N*=40 predicted==measured)",
         ev=[E("cpu_numerical", "model:biot", "author:D"),
             E("analytic", "model:biot", "author:D")],
         probes=["gpu_microbench", "indep_reimpl"]),
    dict(id="CABLE_EXCITABLE", kind="measured_cell", cites=["TRANSFER_LAW", "F_CAPACITY"],
         desc="thread-7 #8 cable/excitable: 3 decorrelated cells (re-run + recompute + "
              "honest-negative companion), T3 cross-cell fingerprint",
         ev=[E("cpu_numerical", "model:cable", "author:D"),
             E("cpu_numerical", "model:cable", "author:peer"),
             E("cpu_numerical", "model:cable", "author:D")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="CYCLIC_NETWORKS", kind="measured_cell", cites=["U5_IDENT_TRAFFIC", "TRANSFER_LAW"],
         desc="thread-7 #9 cyclic rate-networks x3 (cert + identifiability + swap); machine-zero "
              "null ABSTAIN; U5 law 15 sig figs",
         ev=[E("cpu_numerical", "model:kingaltman", "author:D"),
             E("cpu_numerical", "model:kingaltman", "author:D"),
             E("cpu_numerical", "model:kingaltman", "author:D")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="POLYMER_CHAIN", kind="measured_cell", cites=["TRANSFER_LAW"],
         desc="thread-7 #10 HP-lattice exact enumeration SUPPORTED (#false-cert=0); ⚠ G-v "
              "fingerprint-predictor over-abstains x5 (safe direction, flagged for audit)",
         ev=[E("cpu_numerical", "model:hplattice", "author:D")],
         probes=["cpu_numerical", "indep_reimpl"]),
    dict(id="HOMOGENIZATION_HS", kind="measured_cell", cites=["TRANSFER_LAW"],
         desc="family-6: Hashin-Shtrikman bounds AS floors, 0-fit, x3 internal over-det (A)",
         ev=[E("cpu_numerical", "model:hs", "author:A"),
             E("analytic", "model:hs", "author:A")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="STRUCT_FORMATION", kind="modeled_only", cites=["TRANSFER_LAW"],
         desc="family-5 Rosenthal map: MAP-CONSISTENT only (A self-caught the too-clean green); "
              "genuine external validation (FEM latent-heat / NIST AM-Bench melt pools) OWED",
         ev=[E("analytic", "model:rosenthal", "author:A")],
         probes=["external_lit", "cpu_numerical"]),
    dict(id="FSI_CONDUIT", kind="in_flight",
         cites=["TRANSFER_LAW", "FED_FLOOR", "COUPLED_2WAY"],
         desc="family-4 FSI flexible-conduit (claimed on stream; the 2-way coupling instance)",
         ev=[],
         probes=["cpu_numerical", "analytic"]),
    dict(id="MURRAY_BRANCH", kind="open", cites=["TRANSFER_LAW"],
         desc="family-7 transport/branching networks (Murray's law analytic anchor) — unbuilt",
         ev=[],
         probes=["cpu_numerical", "analytic", "external_lit"]),
    dict(id="OOD_TWIN", kind="open", cites=["F_FISHER", "R3_IDENT", "TRANSFER_LAW"],
         desc="genuine new-regime OOD twin test NON-CONCLUSIVE (C: corpus is pure interpolation) "
              "— needs HELD-OUT-REGIME data; deployment rule = ABSTAIN outside training support",
         ev=[E("cpu_numerical", "model:twin", "author:C")],
         probes=["external_lit", "cpu_numerical"]),
    # ================= §H CIRCUIT / PCB / ASIC DESCENT =================
    dict(id="PCB_MAP", kind="measured_cell",
         cites=["F_PEBBLE", "F_FISHER", "R2_DET", "R4_CONT"],
         desc="PCB cert-vector map DONE (facet-by-facet, load-bearing citations verified, "
              "Elmore timing cell SHIP/DON'T-SHIP 168ps/f_max=193MHz closed form)",
         ev=[E("cpu_numerical", "model:pcb", "author:D"),
             E("external_lit"),
             E("analytic", "model:elmore", "author:D")],
         probes=["indep_reimpl", "external_repo"]),
    dict(id="PCB_R1_ROUTE", kind="measured_cell",
         cites=["F_PEBBLE", "PCB_MAP", "U1_MOVE_INFO"],
         desc="PCB R1 routing bound DONE: L* exact via TWO decorrelated instruments "
              "(Thompson-cut + slot-matching), certifying gate holds",
         ev=[E("cpu_numerical", "model:thompson", "author:D"),
             E("cpu_numerical", "model:matching", "author:D")],
         probes=["external_repo", "indep_reimpl"]),
    dict(id="VIA_AWARE_BOUND", kind="open", cites=["PCB_R1_ROUTE"],
         desc="OPEN honest finding: via-pillar capacity binds, INVISIBLE to wire-capacity cuts "
              "(C rejected a tautological attempt — needs a real non-tautological bound)",
         ev=[E("cpu_numerical", "model:thompson", "author:D")],
         probes=["cpu_numerical", "external_lit", "indep_reimpl"]),
    dict(id="PCB_R3_STACKUP", kind="open", cites=["PCB_MAP", "R3_IDENT"],
         desc="OPEN: stackup-param identifiability from edge S-params (twin Fisher)",
         ev=[],
         probes=["cpu_numerical", "analytic"]),
    dict(id="PCB_R4_CONGESTION", kind="open", cites=["PCB_MAP", "R4_CONT"],
         desc="OPEN: congestion R4 on a real netlist (structural vs data-dependent SSN)",
         ev=[],
         probes=["cpu_numerical", "external_repo"]),
    dict(id="ASIC_ANCHOR", kind="measured_cell", cites=["PCB_R1_ROUTE", "F_PEBBLE"],
         desc="ASIC silicon-rung anchor DONE: sky130 HPWL 1.348x on tool-truth "
              "(kill-gate fired + resolved honestly)",
         ev=[E("external_repo", "author:D")],
         probes=["external_repo", "cpu_numerical"]),
    dict(id="ASIC_MINCUT", kind="in_flight",
         cites=["ASIC_ANCHOR", "PCB_MAP", "U1_MOVE_INFO"],
         desc="IN-FLIGHT (burst): §H descent next rung — mincut/partition-based P&R bound toward "
              "certified layout at the routing/timing floor on the open PDK",
         ev=[],
         probes=["external_repo", "cpu_numerical"]),
    # ================= §F INVERSE STACK =================
    dict(id="RENDER_MATCH", kind="measured_cell", cites=["F_FISHER", "R3_IDENT"],
         desc="render-match sigma_min + shuffle-null template (the universal recovery cert)",
         ev=[E("cpu_numerical", "model:fisher", "author:D"),
             E("cpu_numerical", "model:fisher", "author:D")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="MULTIMODAL_SIGMA", kind="measured_cell", cites=["RENDER_MATCH", "U3_FISHER_MI"],
         desc="multi-modal sigma_min (stacked Fisher; independent GRADIENTS not labels; N-way "
              "joint null not pairwise) — H, 4/4",
         ev=[E("cpu_numerical", "model:fisher", "author:H")],
         probes=["cpu_numerical", "external_lit"]),
    dict(id="CONSERVATION_FILL", kind="measured_cell", cites=["RENDER_MATCH", "R3_IDENT"],
         desc="physics-informed fill-in: same-law cert silently false-accepts on the balance "
              "operator's null mode; sigma_min gate = drop-in rule (H, 4/4)",
         ev=[E("cpu_numerical", "model:fisher", "author:H")],
         probes=["cpu_numerical"]),
    dict(id="SPLAT_CERT", kind="in_flight",
         cites=["RENDER_MATCH", "MULTIMODAL_SIGMA", "CONSERVATION_FILL", "F_GEOM_RES"],
         desc="IN-FLIGHT (burst): certified splat {grounded|filled|unknown} — QC-d PASS "
              "(diversity leg kills billboard 7->0) but P1-P4 fail on UNDER-CONVERGED phase-1; "
              "converged re-run owed",
         ev=[E("cpu_numerical", "model:splat", "author:D")],
         probes=["gpu_microbench", "cpu_numerical"]),
    dict(id="ACTIVE_VISION", kind="in_flight", cites=["OED_ENGINE", "RENDER_MATCH"],
         desc="IN-FLIGHT (burst): camera NBV honest-negative reproduced x2 bit-identical (one "
              "effective instrument); adjudication owed: genuine domain-boundary vs "
              "under-developed cell",
         ev=[E("cpu_numerical", "model:oed", "author:D")],
         probes=["cpu_numerical", "indep_reimpl"]),
    dict(id="SUPERRES_CERT", kind="measured_cell",
         cites=["RENDER_MATCH", "F_GEOM_RES", "MULTIMODAL_SIGMA"],
         desc="certified super-resolution on REAL images (C x3: two-way null-space, three-way "
              "{grounded|filled|unknown} w/ epistemic demotion, video motion-lift w/ boxcar "
              "forward-null) — same author+instrument x3 (correlated)",
         ev=[E("cpu_numerical", "model:nullspace", "author:C"),
             E("cpu_numerical", "model:nullspace", "author:C"),
             E("cpu_numerical", "model:nullspace", "author:C")],
         probes=["indep_reimpl", "external_lit", "gpu_microbench"]),
    dict(id="HELMHOLTZ_SCATTER", kind="measured_cell", cites=["RENDER_MATCH", "R3_IDENT"],
         desc="Helmholtz multiscatter forward Bessel-validated; identifiability = geometry (+) "
              "forward-physics (naive diffraction bound false-abstains x18) — C, 6/6",
         ev=[E("cpu_numerical", "model:helmholtz", "author:C")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="GS_12GB", kind="measured_cell", cites=["ENVELOPE", "F_PEBBLE"],
         desc="3DGS-12GB recipe (OOM boundary validated 5-12%, budget-flip measured, fp16-pos "
              "NaN edge); ⚠ quality column MODELED (reference-CUDA leg owed)",
         ev=[E("gpu_microbench", "gpu:local", "author:D"),
             E("analytic", "model:memmodel", "author:D")],
         probes=["external_repo", "gpu_microbench"]),
    # ================= §G GENERATIVE DESIGN =================
    dict(id="GEN_DESIGN", kind="measured_cell",
         cites=["R3_IDENT", "R5_GEOMETRY", "U4_TROPICAL"],
         desc="certified generative design (DfC 53.7x @ +30.2%, gaming law measured; R3 "
              "single-load ceiling CLOSED by C's multi-load span identifiability+discrimination)",
         ev=[E("cpu_numerical", "model:simp", "author:D"),
             E("cpu_numerical", "model:fisher", "author:C")],
         probes=["cpu_numerical", "indep_reimpl"]),
    # ================= §E HARDWARE + REMAINING OPEN =================
    dict(id="ENVELOPE", kind="measured_cell", cites=["ROOFLINE", "R1_TRAFFIC"],
         desc="hardware-envelope flips 12/12 (RTX5070 MEASURED; other silicon spec-MODELED)",
         ev=[E("cpu_numerical", "model:envelope", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_second_device", "external_lit"]),
    dict(id="MULTI_HW_ENSEMBLE", kind="open", cites=["ENVELOPE", "MULTIGPU_FED"],
         desc="§E GAP: ensemble measured across >=2 REAL hardware targets (A100/GTX1650 are "
              "spec-modeled only); selection-rule certified per hardware",
         ev=[E("cpu_numerical", "model:envelope", "author:D")],
         probes=["gpu_second_device", "external_repo"]),
    dict(id="MULTIGPU_FED", kind="open", cites=["ENVELOPE", "F_CAPACITY", "U1_MOVE_INFO"],
         desc="multi-GPU federation floor (NVLink surface/volume; needs real 2nd GPU)",
         ev=[E("analytic", "model:envelope", "author:D")],
         probes=["gpu_second_device", "analytic"]),
    dict(id="NONLOCAL_DAG", kind="open", cites=["F_PEBBLE"],
         desc="non-local general-DAG exact no-recompute floor (PSPACE — bracket only)",
         ev=[],
         probes=["analytic", "cpu_numerical"]),
]


# ----------------------------------------------------------------------------------
# PATCH THE ENGINE'S INSTANCE (data refresh; engine code untouched)
# ----------------------------------------------------------------------------------
def install_instance(nodes):
    node_by_id = {n["id"]: n for n in nodes}
    assert len(nodes) == len(node_by_id), "duplicate node id"
    for n in nodes:
        for c in n["cites"]:
            assert c in node_by_id, f"dangling cite {c} in {n['id']}"
    cited_by = defaultdict(list)
    for n in nodes:
        for c in n["cites"]:
            cited_by[c].append(n["id"])
    for n in nodes:
        n["weight"] = 1.0 + len(cited_by[n["id"]])
        n["prior"] = PRIOR_BY_KIND[n["kind"]]
    eng.NODES = nodes
    eng.NODE_BY_ID = node_by_id
    eng.CITED_BY = cited_by
    return node_by_id, cited_by


def submodularity_and_greedy_gap_v2(seed=7):
    """v1's protocol re-run on a v2 sub-instance (frontier-heavy mix)."""
    rng = random.Random(seed)
    sub_nodes = ["MULTI_STITCH", "ASIC_MINCUT", "AVBD_ROOFLINE", "VIA_AWARE_BOUND",
                 "R2_DET", "F_PEBBLE", "U4_TROPICAL", "TRANSFER_LAW", "SPLAT_CERT",
                 "MULTI_HW_ENSEMBLE", "PCB_MAP", "OED_CALIBRATION"]
    cands = [p for p in eng.build_candidates() if p["node"] in sub_nodes]
    per_node = defaultdict(list)
    for p in cands:
        per_node[p["node"]].append(p)
    trimmed = []
    for nid in sub_nodes:
        trimmed.extend(per_node[nid][:2])
    trimmed = trimmed[:14]
    res = {"n_nodes": len(sub_nodes), "n_candidates": len(trimmed)}
    violations, worst, checks = 0, 0.0, 0
    for _ in range(3000):
        pool = trimmed[:]
        rng.shuffle(pool)
        x, rest = pool[0], pool[1:]
        B = rest[:rng.randint(1, min(6, len(rest)))]
        A = [p for p in B if rng.random() < 0.5]
        gA = eng.marginal_gain(x, A, RHO_CAT)
        gB = eng.marginal_gain(x, B, RHO_CAT)
        checks += 1
        if gB > gA + 1e-12:
            violations += 1
            worst = max(worst, gB - gA)
    res["diminishing_returns_checks"] = checks
    res["violations"] = violations
    res["worst_violation"] = worst
    k = 4
    best_g, best_set = -1.0, None
    for combo in itertools.combinations(trimmed, k):
        g = eng.batch_gain(list(combo), RHO_CAT)
        if g > best_g:
            best_g, best_set = g, combo
    batch, _, g_greedy, _ = eng.greedy_select(trimmed, budget=1e9, rho_cat=RHO_CAT,
                                              max_picks=k, by_cost_ratio=False)
    res["cardinality_k4"] = dict(
        exhaustive_opt=best_g, greedy=g_greedy, ratio=g_greedy / best_g,
        opt_set=[f"{p['node']}x{p['instr']}" for p in best_set],
        greedy_set=[f"{p['node']}x{p['instr']}" for p in batch])
    budget = 25
    best_g, best_set = -1.0, ()
    for r in range(0, len(trimmed) + 1):
        for combo in itertools.combinations(trimmed, r):
            if sum(p["cost"] for p in combo) > budget:
                continue
            g = eng.batch_gain(list(combo), RHO_CAT)
            if g > best_g:
                best_g, best_set = g, combo
    batch, _, g_greedy, _ = eng.greedy_select(trimmed, budget=budget, rho_cat=RHO_CAT,
                                              max_picks=99)
    res["knapsack_b25"] = dict(
        exhaustive_opt=best_g, greedy=g_greedy, ratio=g_greedy / best_g,
        opt_set=[f"{p['node']}x{p['instr']}" for p in best_set],
        greedy_set=[f"{p['node']}x{p['instr']}" for p in batch])
    return res


# The currently-in-flight burst (prospective calibration surface, fixed pre-run):
BURST = {
    "multicomponent-stitch": "MULTI_STITCH",
    "AVBD": "AVBD_ROOFLINE",
    "coupled-2way": "COUPLED_2WAY",
    "splat-rerun": "SPLAT_CERT",
    "hyperelastic-fix": "HYPERELASTIC",
    "active-vision-adjudication": "ACTIVE_VISION",
    "ASIC-mincut": "ASIC_MINCUT",
    "superres": "SUPERRES_CERT",
}
I_SET = ["MULTI_STITCH", "COUPLED_2WAY", "FED_FLOOR", "CORE2_ATTR", "INTEG_STITCH",
         "U4_TROPICAL", "PER_VAR_PRECISION", "KNOB_DERIVE", "AUTONOMY_LOOP"]


def main():
    node_by_id, cited_by = install_instance(NODES_V2)
    evidence = {"script": os.path.basename(__file__),                 "what": "§J node-list refresh (data-only; engine code unchanged, imported)",
                "n_nodes": len(NODES_V2),
                "model_params": dict(P0=eng.P0, P_PRIV=eng.P_PRIV, RHO_CAT=RHO_CAT,
                                     PRIOR_BY_KIND=PRIOR_BY_KIND,
                                     INSTRUMENT_COST=eng.INSTRUMENT_COST),
                "calibration_context": (
                    "retrospective calibration INCONCLUSIVE (noisy outcome "
                    "labels; per-probe realized-uncertainty-reduction not logged). This file's "
                    "prospective engine-vs-burst comparison is the replacement instrument.")}

    evidence["nodes"] = [
        dict(id=n["id"], kind=n["kind"], weight=n["weight"], prior=n["prior"],
             n_evidence=len(n["ev"]), cited_by=cited_by[n["id"]],
             fa_current=round(eng.node_fa(n, [], RHO_CAT), 6),
             undetected_mass=round(n["weight"] * n["prior"] * eng.node_fa(n, [], RHO_CAT), 6),
             desc=n["desc"])
        for n in NODES_V2]
    total_mass = sum(x["undetected_mass"] for x in evidence["nodes"])
    evidence["total_undetected_failure_mass"] = round(total_mass, 6)
    evidence["floor_law_verification"] = eng.verify_floor_law()

    evidence["scenarios"] = {}
    for name, sc in SCENARIOS.items():
        cands = eng.build_candidates(exclude=sc["exclude"], gpu_cost_mult=sc["gpu_cost_mult"])
        batch, trace, g_total, spent = eng.greedy_select(
            cands, sc["budget"], RHO_CAT, max_picks=BATCH_PICKS)
        ranking = eng.full_ranking(cands, RHO_CAT)
        evidence["scenarios"][name] = dict(
            budget=sc["budget"], excluded=sorted(sc["exclude"]),
            batch_top12=trace,
            batch_total_gain=round(g_total, 6), batch_cost=spent,
            fraction_of_mass_resolved=round(g_total / total_mass, 4),
            distinct_instrument_classes_in_batch=sorted({p["instr"] for p in batch}),
            full_ranking_top20=[(nm, round(g, 6)) for nm, g in ranking[:20]],
        )

    evidence["submodularity"] = submodularity_and_greedy_gap_v2()
    evidence["stability"] = eng.stability_analysis()

    # -------------------- PRE-REGISTERED V2 ANCHORS --------------------
    anchors = {}
    grich = eng.build_candidates(gpu_cost_mult=SCENARIOS["gpu_rich"]["gpu_cost_mult"])
    rank_full = eng.full_ranking(grich, RHO_CAT)
    rmap = eng.rank_map(rank_full)
    gains = dict(rank_full)
    all_batches = {sc: [t["node"] + "x" + t["instr"]
                        for t in evidence["scenarios"][sc]["batch_top12"]]
                   for sc in SCENARIOS}

    # V2-A1 stale purge
    stale_hits = [(sc, k) for sc, keys in all_batches.items() for k in keys
                  if k.startswith("PCB_MAPx") or k.startswith("PCB_R1_ROUTEx")]
    stale_best_rank = min([r for k, r in rmap.items()
                           if k.startswith("PCB_MAPx") or k.startswith("PCB_R1_ROUTEx")])
    anchors["V2_A1_stale_pcb_purged"] = dict(
        passed=not stale_hits, offending=stale_hits,
        best_full_ranking_rank_of_any_pcb_probe=stale_best_rank)

    # V2-A2 rho-floor on the settled R2 node
    key_r2rep = "R2_DETxgpu_microbench"
    top_gains = sorted((t["marginal_gain"]
                        for t in evidence["scenarios"]["gpu_rich"]["batch_top12"]), reverse=True)
    med_top = top_gains[len(top_gains) // 2]
    a2_ratio = gains[key_r2rep] / med_top
    anchors["V2_A2_rho_floor_binds"] = dict(
        passed=bool(a2_ratio < 0.05), r2_repeat_gain=round(gains[key_r2rep], 8),
        median_batch_gain=round(med_top, 6), ratio=round(a2_ratio, 6))

    # V2-A3 no same-instrument repeat on >=3-evidence nodes
    bad_picks = []
    for sc in SCENARIOS:
        for t in evidence["scenarios"][sc]["batch_top12"]:
            if t["is_repeat"] and len(node_by_id[t["node"]]["ev"]) >= 3:
                bad_picks.append((sc, t["node"], t["instr"]))
    anchors["V2_A3_no_repeat_on_settled"] = dict(passed=not bad_picks,
                                                 offending_picks=bad_picks)

    # V2-A4 greedy gap
    sm = evidence["submodularity"]
    a4 = min(sm["cardinality_k4"]["ratio"], sm["knapsack_b25"]["ratio"])
    anchors["V2_A4_greedy_gap"] = dict(
        passed=bool(a4 >= 0.95), min_ratio=round(a4, 5),
        dr_violations=sm["violations"], dr_checks=sm["diminishing_returns_checks"])

    # V2-A5 stability
    st = evidence["stability"]
    a5 = (st["both"]["median_spearman"] >= 0.8 and st["both"]["median_top_jaccard"] >= 0.6)
    dominating = ("rho" if st["rho_only"]["median_spearman"] <
                  st["weights_only"]["median_spearman"] else "weights")
    anchors["V2_A5_stability"] = dict(
        passed=bool(a5),
        median_spearman_both=round(st["both"]["median_spearman"], 4),
        median_jaccard_both=round(st["both"]["median_top_jaccard"], 4),
        min_jaccard_both=round(st["both"]["min_top_jaccard"], 4),
        dominating_input=dominating)

    # V2-A6 §I agreement — REPORT-ONLY (not forced, not gating)
    i_probes_in_batch = [k for k in all_batches["gpu_rich"]
                         if k.split("x")[0] in I_SET]
    i_best_ranks = {nid: min([r for k, r in rmap.items() if k.split("x")[0] == nid],
                             default=None) for nid in I_SET}
    anchors["V2_A6_section_I_agreement_REPORT_ONLY"] = dict(
        report_only=True,
        agrees=len(i_probes_in_batch) >= 3,
        i_probes_in_gpu_rich_batch=i_probes_in_batch,
        best_full_ranking_rank_per_I_node=i_best_ranks,
        note="disagreement is a FINDING about value-weighting (propagation-degree), not forced")

    evidence["preregistered_anchors"] = anchors
    gating = {k: a for k, a in anchors.items() if not a.get("report_only")}
    evidence["verdict"] = dict(
        all_gating_anchors_passed=all(a["passed"] for a in gating.values()),
        anchor_summary={k: a.get("passed", "report-only") for k, a in anchors.items()})

    # -------------------- PROSPECTIVE CALIBRATION: engine vs the in-flight burst -------------
    burst_cmp = {}
    for label, nid in BURST.items():
        picked_in = [sc for sc, keys in all_batches.items()
                     if any(k.split("x")[0] == nid for k in keys)]
        node_keys = {k: r for k, r in rmap.items() if k.split("x")[0] == nid}
        best_key = min(node_keys, key=node_keys.get) if node_keys else None
        burst_cmp[label] = dict(
            node=nid, engine_picked_in_scenarios=picked_in,
            would_have_picked=bool(picked_in),
            best_probe=best_key,
            best_full_ranking_rank=node_keys.get(best_key),
            best_probe_gain=round(gains.get(best_key, 0.0), 6))
    evidence["prospective_calibration_burst_comparison"] = burst_cmp

    evidence["declared_limits"] = [
        "same declared limits as v1 (cross-node common causes outside the factorized objective; "
        "priors are judgment inputs; rho_cat engineering estimates priced by A5)",
        "the node instance is a hand-transcription of the graph and audit state; "
        "evidence counts are per-instrument-construction, not per-run (bit-identical "
        "re-runs of the same construction counted once or tagged same-model)",
        "MULTI_STITCH is a SINK node (nothing cites the north-star) so propagation-degree gives "
        "it weight 1; its enablers carry the propagated value — V2-A6 tests exactly this and "
        "reports rather than forces",
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(evidence, f, indent=1)

    # ---- console report ----
    print("=" * 78)
    print(f"OED ENGINE — v2 node instance ({len(NODES_V2)} nodes, "
          f"total undetected-failure mass {total_mass:.3f})")
    print("=" * 78)
    for name in SCENARIOS:
        sc = evidence["scenarios"][name]
        print(f"\n--- scenario {name} (budget {sc['budget']}, excl {sc['excluded'] or 'none'}) "
              f"— gain {sc['batch_total_gain']} = {100*sc['fraction_of_mass_resolved']:.1f}% "
              f"of mass, cost {sc['batch_cost']} ---")
        for i, t in enumerate(sc["batch_top12"]):
            print(f"  {i+1:2d}. {t['node']:<20s} x {t['instr']:<18s} "
                  f"gain={t['marginal_gain']:.4f} cost={t['cost']} "
                  f"g/c={t['gain_per_cost']:.4f}" + ("  [REPEAT]" if t["is_repeat"] else ""))
    print("\n--- submodularity / greedy gap (v2 sub-instance) ---")
    print(f"  DR violations: {sm['violations']}/{sm['diminishing_returns_checks']} "
          f"(worst {sm['worst_violation']:.2e})")
    print(f"  cardinality k=4: greedy/opt = {sm['cardinality_k4']['ratio']:.5f}")
    print(f"  knapsack b=25 : greedy/opt = {sm['knapsack_b25']['ratio']:.5f}")
    print("\n--- stability (engine protocol) ---")
    for mode in ("rho_only", "weights_only", "both"):
        s = st[mode]
        print(f"  {mode:<13s} spearman med={s['median_spearman']:.3f} "
              f"min={s['min_spearman']:.3f}  jaccard med={s['median_top_jaccard']:.3f} "
              f"min={s['min_top_jaccard']:.3f}")
    print("\n--- PRE-REGISTERED V2 ANCHORS ---")
    for k, a in anchors.items():
        tag = "REPORT" if a.get("report_only") else ("PASS" if a["passed"] else "FAIL")
        print(f"  {tag:<6s} {k}")
    a6 = anchors["V2_A6_section_I_agreement_REPORT_ONLY"]
    print(f"    §I agreement: {a6['agrees']} — I-probes in gpu_rich batch: "
          f"{a6['i_probes_in_gpu_rich_batch']}")
    print("\n--- PROSPECTIVE CALIBRATION (engine vs in-flight burst) ---")
    for label, d in burst_cmp.items():
        print(f"  {'PICKED ' if d['would_have_picked'] else 'not-picked':<10s} {label:<28s} "
              f"best={d['best_probe']} rank={d['best_full_ranking_rank']} "
              f"gain={d['best_probe_gain']}")
    print(f"\nverdict: all_gating_anchors_passed = "
          f"{evidence['verdict']['all_gating_anchors_passed']}")
    print(f"evidence -> {OUT}")


if __name__ == "__main__":
    main()
