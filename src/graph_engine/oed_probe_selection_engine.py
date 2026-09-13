#!/usr/bin/env python3
"""
d_oed_probe_selection_engine.py — §5's missing organ made executable: the self-improving
loop's node-selection step as BATCH-OED SENSOR PLACEMENT on the actual MASTER_UNLOCK_GRAPH.

FRAMING (operator): probes are SENSORS placed on the unknown graph; redundancy must buy
DECORRELATION, not repetition. Formalized here, derived not vibed:

THE MODEL
  * Node truth = latent Bernoulli: node v is materially WRONG with prior pi_v (by claim
    kind: theorem-shaped / measured-cell / modeled-only / in-flight / open).
  * Evidence item = noisy sensor: catches a wrong node unless it MISSES. Misses are
    CORRELATED through shared failure-mode TAGS (same instrument-class, same GPU, same
    analytic model, same author-cell) — the multi-common-cause (CCF beta-factor) model:
        sensor i misses  iff  (any shared common-cause event C_t fires, t in tags(i))
                              OR its private miss event I_i fires,
        P(C_t) = rho_t * P0,   P(I_i) = p_priv.
    Joint miss FA(E) = P(all sensors in E miss | node wrong) computed EXACTLY by
    enumeration over the 2^T tag-event outcomes (no p^N fiction). This reproduces the
    repo's ρ-floor law: for 2 sensors  P(both miss) = p̄² + ρ_ind·p̄(1−p̄) (verified
    numerically below as an identity in the measured miss-indicator correlation), and
    for k same-class repeats  P(all miss) = q + (1−q)·p_priv^k  →  floors at q = the
    common-cause rate: a same-instrument repeat buys ~NOTHING once the floor binds
    (β-clamp: FA ≥ β·p — the 'winning correction' per the inherited law).
  * Value weight w_v = 1 + (#nodes citing v)  — propagation-degree from the map's own
    cross-reference (cites) lists, hand-transcribed from MASTER_UNLOCK_GRAPH §1/§3/§6.

THE OBJECTIVE (what a probe batch buys)
  Undetected-failure mass  M(E) = Σ_v  w_v · pi_v · FA(E_v)
    = expected propagation-weighted probability that a node is wrong AND every sensor
      on it missed (the silent-poison mass). A probe that FIRES converts a
      silent failure into a known fix-task, so the batch's information gain is
        G(S) = Σ_v w_v · pi_v · [ FA(E_v) − FA(E_v ∪ S_v) ]  ≥ 0, monotone.
  (P(wrong | all passed) = pi·FA / (pi·FA + (1−pi)) is monotone in pi·FA, so ranking by
   G is ranking by posterior-failure reduction; both are emitted.)

SUBMODULARITY — checked, not cited: D-optimal batch selection is submodular for
linear-Gaussian sensors; this posterior-Bernoulli/common-cause variant is NOT covered by
that theorem (the tag-event structure could create complementarity). We therefore
(1) MEASURE the greedy gap by exhaustive-vs-greedy on a ≤12-node sub-instance
    (cardinality k=4 AND knapsack-budget variants), and
(2) directly sample the diminishing-returns inequality g(x|A) ≥ g(x|B) for A ⊆ B.

KNOWN LIMIT (pairwise decorrelation does not imply joint coverage): the objective factorizes per node, so CROSS-NODE common causes (the same
local GPU lying to many nodes at once) are not in the objective; the engine reports the
batch's instrument-class diversity as the N-way-co-miss mitigation surface, it does not
certify joint coverage.

PRE-REGISTERED VERDICT CRITERIA (mandatory, §6; fixed before first run):
  A1  In the GPU-rich full ranking, probe (R4_CONT × gpu_microbench) — a DECORRELATED
      real instrument on a 1-evidence modeled-only load-bearing node — ranks ABOVE
      (R2_DET × gpu_microbench) — a 5th same-instrument repeat of the settled
      determinism 4-branch node; and the R2 repeat appears in NO scenario's top-10.
  A2  Quantitative ρ-floor bind: marginal gain of the 5th same-instrument R2 repeat
      < 5% of the median top-10 marginal gain (GPU-rich).
  A3  No scenario's top-10 contains a same-instrument repeat on any node with ≥3
      existing evidence items (repetition never beats decorrelation anywhere).
  A4  Greedy/exhaustive optimal ratio ≥ 0.95 on the sub-instance (else: measured gap
      reported, engine flagged non-submodular-in-practice).
  A5  Stability (SYMMETRIC-QC): under ρ ±50% and weights ×/÷2 perturbations, median
      Spearman rank-correlation of the full ranking ≥ 0.8 AND median top-10 Jaccard
      ≥ 0.6; else the engine is 'not yet a tool' — reported honestly with the
      dominating input identified.
Failure of A1–A3 means either the engine is wrong (fix) or the human intuition is
(report the surprise). No post-hoc threshold moves.

Run: python3 e.py
CPU-only, deterministic (seeded). Evidence JSON →
scripts/physics_exp/artifacts/d_oed_probe_selection_evidence.json
Author: author,.
"""
import itertools
import json
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts", "d_oed_probe_selection_evidence.json")

# ----------------------------------------------------------------------------------
# 1. FAILURE-CORRELATION MODEL PARAMETERS (the honest inputs; perturbed in QC)
# ----------------------------------------------------------------------------------
P0 = 0.15        # base common-cause scale: P(C_t) = rho_cat(tag) * P0
P_PRIV = 0.10    # private miss probability of any single sensor
# tag-category correlation strengths (rho): how strongly a shared tag couples misses
RHO_CAT = {
    "instr": 0.45,   # same instrument class (systematic instrument blind spot)
    "gpu": 0.35,     # same physical GPU (same silicon/driver lying the same way)
    "model": 0.50,   # same analytic model family (shared modeling assumption)
    "author": 0.15,  # same author-cell (shared construction/expectation bias)
}
PRIOR_BY_KIND = {  # P(claim materially wrong BEFORE the listed evidence)
    "theorem_shaped": 0.35,  # analytic claim, could hide a scope error
    "measured_cell": 0.30,   # empirical claim from a built cell
    "modeled_only": 0.50,    # claim exists but only as a model, never real-tested
    "in_flight": 0.55,       # being built this window, unresolved
    "open": 0.70,            # unbuilt next-wave material
}
INSTRUMENT_COST = {  # cost units ~ agent-effort; GPU probes carry GPU-minutes
    "analytic": 2, "cpu_numerical": 3, "external_lit": 4, "indep_reimpl": 6,
    "gpu_microbench": 8, "external_repo": 10, "gpu_second_device": 25,
}
GPU_INSTRUMENTS = {"gpu_microbench", "gpu_second_device"}

SCENARIOS = {
    # gpu_rich: GPU-minutes are ABUNDANT -> GPU instrument cost halved (the cost is
    # the scarce currency; abundance changes the price, not just the permission)
    "gpu_rich": {"budget": 60, "exclude": set(), "gpu_cost_mult": 0.5},
    "cpu_only": {"budget": 40, "exclude": GPU_INSTRUMENTS, "gpu_cost_mult": 1.0},
    "one_agent": {"budget": 12, "exclude": set(), "gpu_cost_mult": 1.0},
}

# ----------------------------------------------------------------------------------
# 2. THE GRAPH INSTANCE — hand-transcribed from MASTER_UNLOCK_GRAPH.md §1+§3+§6 (+§7)
# ev = existing evidence items, each a list of shared failure-mode tags.
# cites = which nodes this node's claim DEPENDS ON (map cross-references);
# w_v = 1 + #citing(v) is derived below.
# probes = applicable instrument classes for a NEXT probe (incl. deliberate repeats).
# ----------------------------------------------------------------------------------
E = lambda instr, *extra: [f"instr:{instr}", *extra]  # noqa: E731
NODES = [
    # --- §1 floors ---
    dict(id="F_LANDAUER", kind="measured_cell", cites=[],
         desc="Landauer kT ln2 tight-to-1%, Jarzynski-anchored (leg A4)",
         ev=[E("cpu_numerical", "model:jarzynski", "author:D"),
             E("analytic", "model:jarzynski", "author:D")],
         probes=["external_lit", "indep_reimpl", "cpu_numerical"]),
    dict(id="F_PEBBLE", kind="measured_cell", cites=[],
         desc="Hong-Kung red-blue pebble floor on real kernels (zoo 6/8, C529, float4)",
         ev=[E("gpu_microbench", "gpu:local", "author:D", "model:hk"),
             E("gpu_microbench", "gpu:local", "author:D", "model:hk"),
             E("gpu_microbench", "gpu:local", "author:D"),
             E("analytic", "model:hk", "author:D")],
         probes=["gpu_second_device", "external_lit", "indep_reimpl", "gpu_microbench"]),
    dict(id="F_CAPACITY", kind="in_flight", cites=[],
         desc="Shannon capacity / signal-energy floor (PCIe 39x/byte measured)",
         ev=[E("gpu_microbench", "gpu:local", "author:D")],
         probes=["analytic", "external_lit", "cpu_numerical", "gpu_microbench"]),
    dict(id="F_FISHER", kind="measured_cell", cites=[],
         desc="Fisher/CRB floor, CRB ratio 1.08, chi-threshold proven adversarially",
         ev=[E("cpu_numerical", "model:fisher", "author:D"),
             E("analytic", "model:fisher", "author:D"),
             E("cpu_numerical", "model:fisher", "author:D")],
         probes=["external_lit", "indep_reimpl"]),
    dict(id="F_TUR", kind="measured_cell", cites=["F_LANDAUER"],
         desc="TUR precision-per-dissipation achievable (Q->1.006), two model classes",
         ev=[E("cpu_numerical", "model:langevin", "author:D"),
             E("cpu_numerical", "model:langevin", "author:D")],
         probes=["external_lit", "analytic", "indep_reimpl"]),
    # --- §1 unification points (theorem-shaped) ---
    dict(id="U1_MOVE_INFO", kind="theorem_shaped", cites=["F_PEBBLE", "F_CAPACITY"],
         desc="U1: Hong-Kung min-traffic IS an information bound; bytes x E_bit",
         ev=[E("analytic", "model:hk", "author:D")],
         probes=["indep_reimpl", "external_lit", "gpu_microbench"]),
    dict(id="U2_ERASE_TRANSPORT", kind="theorem_shaped", cites=["F_LANDAUER", "F_CAPACITY"],
         desc="U2: Landauer kTln2 vs Shannon-at-kT — same constant, different scope",
         ev=[E("analytic", "author:D")],
         probes=["external_lit", "analytic", "cpu_numerical"]),
    dict(id="U3_FISHER_MI", kind="theorem_shaped", cites=["F_FISHER", "R3_IDENT"],
         desc="U3: Fisher = local mutual information; chi-threshold unifies R3/twin/sensor",
         ev=[E("analytic", "model:fisher", "author:D")],
         probes=["indep_reimpl", "external_lit", "cpu_numerical"]),
    dict(id="U4_TROPICAL", kind="theorem_shaped", cites=[],
         desc="U4: cert composition = min-plus lattice, residual decomposes per component",
         ev=[E("analytic", "author:D"), E("analytic", "author:D")],
         probes=["gpu_microbench", "cpu_numerical", "indep_reimpl"]),
    # --- §3 cert-vector facets (compute column = GREEN except R4) ---
    dict(id="R1_TRAFFIC", kind="measured_cell", cites=["F_PEBBLE"],
         desc="R1 traffic cert on real kernels (float4 89%-of-peak certified)",
         ev=[E("gpu_microbench", "gpu:local", "author:D", "model:hk"),
             E("gpu_microbench", "gpu:local", "author:D", "model:hk")],
         probes=["gpu_second_device", "indep_reimpl", "external_lit"]),
    dict(id="R2_DET", kind="measured_cell", cites=[],
         desc="R2 determinism node SETTLED: READ/SOLVE-normal/SOLVE-non-normal/adversarial "
              "(4 branches, all same instrument+GPU+author+model)",
         ev=[E("gpu_microbench", "gpu:local", "author:D", "model:detmode"),
             E("gpu_microbench", "gpu:local", "author:D", "model:detmode"),
             E("gpu_microbench", "gpu:local", "author:D", "model:detmode"),
             E("gpu_microbench", "gpu:local", "author:D", "model:detmode")],
         probes=["gpu_microbench", "indep_reimpl", "external_lit", "cpu_numerical"]),
    dict(id="R3_IDENT", kind="measured_cell", cites=["F_FISHER"],
         desc="R3 QoI identifiability + abstain-on-null-space (chi, adversarial)",
         ev=[E("cpu_numerical", "model:fisher", "author:D"),
             E("analytic", "model:fisher", "author:D")],
         probes=["gpu_microbench", "external_lit"]),
    dict(id="R4_CONT", kind="modeled_only", cites=[],
         desc="R4 contention/addressing cert-kind — MODELED-ONLY; real input-keyed "
              "histogram/sort test is §6 tail #3 (256%-class contention variance)",
         ev=[E("analytic", "model:contention", "author:D")],
         probes=["gpu_microbench", "cpu_numerical", "indep_reimpl"]),
    dict(id="ROOFLINE", kind="measured_cell", cites=[],
         desc="roofline facet: % of BW/FLOP peak measured on this silicon",
         ev=[E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_second_device", "external_lit"]),
    # --- §3 verified cells ---
    dict(id="FP32_ABS", kind="measured_cell", cites=["R2_DET"],
         desc="2^24 fp32-absorption: det-PASS ⊥ correct-FAIL reachable",
         ev=[E("gpu_microbench", "gpu:local", "author:D"), E("analytic", "author:D")],
         probes=["cpu_numerical", "external_lit"]),
    dict(id="AUTONOMY_LOOP", kind="measured_cell",
         cites=["R1_TRAFFIC", "R2_DET", "R3_IDENT", "R4_CONT", "DEPLOY_GATE", "CORE2_ATTR"],
         desc="autonomy loop end-to-end on real nvcc CUDA (weak-gen->cert->refine->89.3%)",
         ev=[E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_microbench", "indep_reimpl", "gpu_second_device"]),
    dict(id="DEPLOY_GATE", kind="measured_cell",
         cites=["R1_TRAFFIC", "R2_DET", "R3_IDENT", "R4_CONT"],
         desc="deployment-gate static SHIP/DON'T-SHIP on real solver source",
         ev=[E("cpu_numerical", "model:static_analysis", "author:D")],
         probes=["gpu_microbench", "indep_reimpl"]),
    dict(id="ENVELOPE", kind="measured_cell", cites=["ROOFLINE", "R1_TRAFFIC"],
         desc="hardware-envelope flips 12/12: best-in-class = f(task, hw)",
         ev=[E("cpu_numerical", "model:envelope", "author:D"),
             E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_second_device", "external_lit"]),
    dict(id="INTEG_STITCH", kind="theorem_shaped", cites=["U4_TROPICAL", "R4_CONT"],
         desc="integration stitch ANALYTIC (binds after 4 seams; binding=contact); "
              "real-GPU run in-flight",
         ev=[E("analytic", "model:stitch", "author:D")],
         probes=["gpu_microbench", "cpu_numerical"]),
    dict(id="CORE2_ATTR", kind="in_flight",
         cites=["R1_TRAFFIC", "R2_DET", "R3_IDENT", "R4_CONT"],
         desc="2c-real static attribution Jacobian — THE load-bearing CORE-2 residual",
         ev=[E("analytic", "author:D")],
         probes=["gpu_microbench", "cpu_numerical"]),
    # --- in-flight rungs ---
    dict(id="LANDAUER_GPU", kind="in_flight", cites=["F_LANDAUER"],
         desc="real-GPU J/bit Landauer residual-ladder rung", ev=[],
         probes=["gpu_microbench", "external_lit"]),
    dict(id="CHANNEL_CELL", kind="in_flight", cites=["F_CAPACITY", "U1_MOVE_INFO"],
         desc="channel-capacity signal-energy floor cell", ev=[],
         probes=["gpu_microbench", "analytic", "external_lit"]),
    dict(id="PCB_MAP", kind="in_flight",
         cites=["F_PEBBLE", "F_FISHER", "R2_DET", "R4_CONT"],
         desc="PCB cert-vector map (5 facets instantiated for circuit/PCB)", ev=[],
         probes=["cpu_numerical", "analytic", "external_lit"]),
    dict(id="PCB_R1_ROUTE", kind="in_flight",
         cites=["F_PEBBLE", "PCB_MAP", "U1_MOVE_INFO"],
         desc="PCB R1 routing lower-bound cell (bisection-width/Thompson area)", ev=[],
         probes=["cpu_numerical", "external_repo", "analytic"]),
    # --- §3 OPEN + §6 tail ---
    dict(id="ASIC_ANCHOR", kind="open", cites=["PCB_R1_ROUTE", "F_PEBBLE"],
         desc="tail#5: reproduce R1 routing bound on OpenROAD/SkyWater PDK", ev=[],
         probes=["external_repo", "external_lit"]),
    dict(id="NONLOCAL_DAG", kind="open", cites=["F_PEBBLE"],
         desc="non-local general-DAG exact floor (PSPACE — bracket only)", ev=[],
         probes=["analytic", "cpu_numerical"]),
    dict(id="OOD_TWIN", kind="open", cites=["F_FISHER", "R3_IDENT"],
         desc="genuine new-regime OOD twin deployment (C's owed test)", ev=[],
         probes=["cpu_numerical", "external_lit", "gpu_microbench"]),
    dict(id="REVERSIBLE_U2", kind="open", cites=["U2_ERASE_TRANSPORT", "F_LANDAUER"],
         desc="tail#6: reversible/adiabatic corner where Landauer != transport floor",
         ev=[], probes=["external_lit", "analytic", "cpu_numerical"]),
    dict(id="MULTIGPU_FED", kind="open",
         cites=["ENVELOPE", "F_CAPACITY", "U1_MOVE_INFO"],
         desc="multi-GPU federation floor (NVLink surface/volume; needs real 2nd GPU)",
         ev=[E("analytic", "model:envelope", "author:D")],
         probes=["gpu_second_device", "analytic"]),
    dict(id="CONTACT_PRIV", kind="open", cites=["INTEG_STITCH", "R4_CONT"],
         desc="tail#1: contact privatization off the 3-4% serialization floor",
         ev=[E("gpu_microbench", "gpu:local", "author:D")],
         probes=["gpu_microbench", "cpu_numerical"]),
    dict(id="LBM_SOA", kind="open", cites=["F_PEBBLE", "R1_TRAFFIC", "ROOFLINE"],
         desc="tail#2: LBM AoS->SoA close 22%->79% roofline gap (R1 floor 72 B/voxel)",
         ev=[E("gpu_microbench", "gpu:local", "author:D"),
             E("analytic", "model:hk", "author:D")],
         probes=["gpu_microbench"]),
    dict(id="FED_FLOOR", kind="open", cites=["U4_TROPICAL", "LBM_SOA", "CONTACT_PRIV"],
         desc="tail#7: composed residual ladder LBM+contact via U4 min-composition",
         ev=[], probes=["cpu_numerical", "analytic"]),
    dict(id="GRAPH_HOLE", kind="open", cites=[],
         desc="tail#8: missing-triangle hole-predictor over this node list (meta-OED)",
         ev=[], probes=["cpu_numerical"]),
]
NODE_BY_ID = {n["id"]: n for n in NODES}
assert len(NODES) == len(NODE_BY_ID), "duplicate node id"
for n in NODES:
    for c in n["cites"]:
        assert c in NODE_BY_ID, f"dangling cite {c} in {n['id']}"

# propagation-degree value weight: w = 1 + #nodes citing v (derived, not assigned)
CITED_BY = defaultdict(list)
for n in NODES:
    for c in n["cites"]:
        CITED_BY[c].append(n["id"])
for n in NODES:
    n["weight"] = 1.0 + len(CITED_BY[n["id"]])
    n["prior"] = PRIOR_BY_KIND[n["kind"]]


# ----------------------------------------------------------------------------------
# 3. THE EXACT JOINT-MISS COMPUTATION (multi-common-cause / beta-factor, enumerated)
# ----------------------------------------------------------------------------------
def tag_rho(tag, rho_cat):
    return rho_cat.get(tag.split(":", 1)[0], 0.0)


def joint_miss(sensors, rho_cat, p0=P0, p_priv=P_PRIV):
    """P(all sensors miss | node wrong). sensors = list of tag-lists. Exact."""
    if not sensors:
        return 1.0  # no sensor ever placed: a wrong node goes undetected surely
    tags = sorted({t for s in sensors for t in s if tag_rho(t, rho_cat) > 0})
    q = [tag_rho(t, rho_cat) * p0 for t in tags]
    idx = {t: i for i, t in enumerate(tags)}
    sensor_masks = []
    for s in sensors:
        m = 0
        for t in s:
            if t in idx:
                m |= 1 << idx[t]
        sensor_masks.append(m)
    total = 0.0
    for assign in range(1 << len(tags)):
        p_assign = 1.0
        for i, qi in enumerate(q):
            p_assign *= qi if (assign >> i) & 1 else (1.0 - qi)
        if p_assign == 0.0:
            continue
        p_all_miss = 1.0
        for m in sensor_masks:
            if not (m & assign):        # no shared cause fired for this sensor
                p_all_miss *= p_priv    # needs its private miss
        total += p_assign * p_all_miss
    return total


def sensor_marginal_miss(sensor_tags, rho_cat, p0=P0, p_priv=P_PRIV):
    keep = 1.0 - p_priv
    for t in sensor_tags:
        keep *= 1.0 - tag_rho(t, rho_cat) * p0
    return 1.0 - keep


# ----------------------------------------------------------------------------------
# 4. CANDIDATE PROBES = (node, instrument-class) pairs
# ----------------------------------------------------------------------------------
def probe_tags(node, instr):
    """Failure-mode tags a NEW probe of this instrument class would carry.
    * A REPEAT (same instrument class already present on the node) re-runs the same
      construction: it inherits the model tags of its same-class predecessors.
    * Same-author analytic/cpu probes inherit the node's dominant model tag
      (shared modeling assumptions).
    * indep_reimpl decorrelates AUTHOR/CONSTRUCTION but not the MODEL CLASS
      (re-deriving the same model shares its blind spot); only a different
      instrument on reality escapes the model tag.
    * external_lit / external_repo are fully external anchors (no shared tags).
    """
    tags = [f"instr:{instr}"]
    node_model_tags = sorted({t for s in node["ev"] for t in s if t.startswith("model:")})
    same_class_model_tags = sorted(
        {t for s in node["ev"] if s[0] == f"instr:{instr}"
         for t in s if t.startswith("model:")})
    if instr == "gpu_microbench":
        tags += ["gpu:local", "author:D"] + same_class_model_tags
    elif instr == "gpu_second_device":
        tags += ["author:D"]  # different silicon, same author-cell
    elif instr in ("cpu_numerical", "analytic"):
        tags += ["author:D"] + sorted(set(node_model_tags[:1] + same_class_model_tags))
    elif instr == "indep_reimpl":
        tags += node_model_tags[:1]
    return tags


def build_candidates(exclude=frozenset(), gpu_cost_mult=1.0):
    cands = []
    for n in NODES:
        for instr in n["probes"]:
            if instr in exclude:
                continue
            existing_instr = {s[0].split(":", 1)[1] for s in n["ev"]}
            cost = INSTRUMENT_COST[instr] * (
                gpu_cost_mult if instr in GPU_INSTRUMENTS else 1.0)
            cands.append(dict(
                node=n["id"], instr=instr, cost=cost,
                tags=probe_tags(n, instr),
                is_repeat=instr in existing_instr,
                n_existing_ev=len(n["ev"]),
            ))
    return cands


# ----------------------------------------------------------------------------------
# 5. OBJECTIVE + GREEDY BATCH SELECTION
# ----------------------------------------------------------------------------------
def node_fa(node, extra_probes, rho_cat):
    sensors = list(node["ev"]) + [p["tags"] for p in extra_probes]
    return joint_miss(sensors, rho_cat)


def batch_gain(batch, rho_cat, weights=None):
    """G(S) = sum_v w_v pi_v (FA(E_v) - FA(E_v u S_v))."""
    by_node = defaultdict(list)
    for p in batch:
        by_node[p["node"]].append(p)
    g = 0.0
    for nid, probes in by_node.items():
        n = NODE_BY_ID[nid]
        w = weights[nid] if weights else n["weight"]
        fa0 = node_fa(n, [], rho_cat)
        fa1 = node_fa(n, probes, rho_cat)
        g += w * n["prior"] * (fa0 - fa1)
    return g


def marginal_gain(probe, batch, rho_cat, weights=None):
    n = NODE_BY_ID[probe["node"]]
    w = weights[probe["node"]] if weights else n["weight"]
    same = [p for p in batch if p["node"] == probe["node"]]
    fa0 = node_fa(n, same, rho_cat)
    fa1 = node_fa(n, same + [probe], rho_cat)
    return w * n["prior"] * (fa0 - fa1)


def greedy_select(cands, budget, rho_cat, weights=None, max_picks=10,
                  by_cost_ratio=True):
    """Cost-benefit greedy; per Khuller/Krause the knapsack guarantee needs
    max(ratio-greedy, best-single) — both computed."""
    batch, spent, trace = [], 0.0, []
    remaining = list(cands)
    while remaining and len(batch) < max_picks:
        best, best_key, best_g = None, -1.0, 0.0
        for p in remaining:
            if spent + p["cost"] > budget:
                continue
            g = marginal_gain(p, batch, rho_cat, weights)
            key = g / p["cost"] if by_cost_ratio else g
            if key > best_key:
                best, best_key, best_g = p, key, g
        if best is None or best_g <= 1e-12:
            break
        batch.append(best)
        spent += best["cost"]
        remaining.remove(best)
        trace.append(dict(node=best["node"], instr=best["instr"], cost=best["cost"],
                          marginal_gain=round(best_g, 6),
                          gain_per_cost=round(best_g / best["cost"], 6),
                          is_repeat=best["is_repeat"]))
    total = batch_gain(batch, rho_cat, weights)
    # best affordable single (knapsack guarantee companion)
    singles = [(marginal_gain(p, [], rho_cat, weights), p) for p in cands
               if p["cost"] <= budget]
    best_single = max(singles, key=lambda x: x[0]) if singles else (0.0, None)
    if best_single[0] > total:
        batch = [best_single[1]]
        trace = [dict(node=batch[0]["node"], instr=batch[0]["instr"],
                      cost=batch[0]["cost"],
                      marginal_gain=round(best_single[0], 6),
                      gain_per_cost=round(best_single[0] / batch[0]["cost"], 6),
                      is_repeat=batch[0]["is_repeat"], note="best-single beat greedy")]
        total = best_single[0]
    return batch, trace, total, spent


def full_ranking(cands, rho_cat, weights=None):
    ranked = sorted(cands, key=lambda p: -marginal_gain(p, [], rho_cat, weights))
    return [(f"{p['node']}x{p['instr']}",
             marginal_gain(p, [], rho_cat, weights)) for p in ranked]


# ----------------------------------------------------------------------------------
# 6. LAW VERIFICATION — the model reproduces the repo's rho-floor functional forms
# ----------------------------------------------------------------------------------
def verify_floor_law():
    out = {}
    # (i) 2-sensor identity: P(both miss) == pbar^2 + rho_ind*pbar*(1-pbar)
    s = ["instr:gpu_microbench", "gpu:local", "author:D", "model:detmode"]
    p1 = sensor_marginal_miss(s, RHO_CAT)
    p11 = joint_miss([s, s], RHO_CAT)
    rho_ind = (p11 - p1 * p1) / (p1 * (1 - p1))
    law = p1 * p1 + rho_ind * p1 * (1 - p1)
    out["two_sensor_identity"] = dict(
        pbar=p1, joint_miss=p11, rho_ind=rho_ind, law_value=law,
        identity_abs_err=abs(law - p11),
        note="algebraic identity once rho is the miss-indicator Pearson corr "
             "(inherited law, body 390); substantive content = tag-driven rho")
    # (ii) k same-class repeats: FA_k -> floor q (beta-clamp FA >= beta*p);
    # vs decorrelated instrument.
    reps = {}
    for k in range(1, 7):
        reps[k] = joint_miss([s] * k, RHO_CAT)
    q_floor = 1.0 - math.prod(1 - tag_rho(t, RHO_CAT) * P0 for t in s)
    decor = joint_miss([s] * 4 + [["instr:indep_reimpl"]], RHO_CAT)
    kish = {k: k / (1 + (k - 1) * rho_ind) for k in (2, 4, 6)}
    out["repeat_floor"] = dict(
        fa_by_repeats={str(k): v for k, v in reps.items()},
        common_cause_floor_q=q_floor,
        fa_4_repeats_plus_one_decorrelated=decor,
        kish_neff_same_class=kish,
        note="FA saturates at the common-cause floor q: the 5th repeat buys "
             f"{reps[5]-reps[6]:.2e} while ONE decorrelated probe buys {reps[4]-decor:.4f}")
    return out


# ----------------------------------------------------------------------------------
# 7. SUBMODULARITY: measured, not assumed
# ----------------------------------------------------------------------------------
def submodularity_and_greedy_gap(seed=7):
    rng = random.Random(seed)
    sub_nodes = ["R4_CONT", "R2_DET", "F_PEBBLE", "U4_TROPICAL", "INTEG_STITCH",
                 "CORE2_ATTR", "PCB_MAP", "ASIC_ANCHOR", "MULTIGPU_FED", "LBM_SOA",
                 "F_CAPACITY", "CHANNEL_CELL"]
    cands = [p for p in build_candidates() if p["node"] in sub_nodes]
    # trim to 14 candidates, keeping diversity (top-2 instruments per node)
    per_node = defaultdict(list)
    for p in cands:
        per_node[p["node"]].append(p)
    trimmed = []
    for nid in sub_nodes:
        trimmed.extend(per_node[nid][:2])
    trimmed = trimmed[:14]
    res = {"n_nodes": len(sub_nodes), "n_candidates": len(trimmed)}

    # (a) direct diminishing-returns sampling: g(x|A) >= g(x|B), A subset of B
    violations, worst, checks = 0, 0.0, 0
    for _ in range(3000):
        pool = trimmed[:]
        rng.shuffle(pool)
        x = pool[0]
        rest = pool[1:]
        b_sz = rng.randint(1, min(6, len(rest)))
        B = rest[:b_sz]
        A = [p for p in B if rng.random() < 0.5]
        gA = marginal_gain(x, A, RHO_CAT)
        gB = marginal_gain(x, B, RHO_CAT)
        checks += 1
        if gB > gA + 1e-12:
            violations += 1
            worst = max(worst, gB - gA)
    res["diminishing_returns_checks"] = checks
    res["violations"] = violations
    res["worst_violation"] = worst

    # (b) cardinality k=4 exhaustive vs plain-gain greedy
    k = 4
    best_g, best_set = -1.0, None
    for combo in itertools.combinations(trimmed, k):
        g = batch_gain(list(combo), RHO_CAT)
        if g > best_g:
            best_g, best_set = g, combo
    batch, _, g_greedy, _ = greedy_select(trimmed, budget=1e9, rho_cat=RHO_CAT,
                                          max_picks=k, by_cost_ratio=False)
    res["cardinality_k4"] = dict(
        exhaustive_opt=best_g, greedy=g_greedy, ratio=g_greedy / best_g,
        opt_set=[f"{p['node']}x{p['instr']}" for p in best_set],
        greedy_set=[f"{p['node']}x{p['instr']}" for p in batch])

    # (c) knapsack budget=25 exhaustive vs (ratio-greedy, best-single) max
    budget = 25
    best_g, best_set = -1.0, ()
    for r in range(0, len(trimmed) + 1):
        for combo in itertools.combinations(trimmed, r):
            if sum(p["cost"] for p in combo) > budget:
                continue
            g = batch_gain(list(combo), RHO_CAT)
            if g > best_g:
                best_g, best_set = g, combo
    batch, _, g_greedy, _ = greedy_select(trimmed, budget=budget, rho_cat=RHO_CAT,
                                          max_picks=99)
    res["knapsack_b25"] = dict(
        exhaustive_opt=best_g, greedy=g_greedy, ratio=g_greedy / best_g,
        opt_set=[f"{p['node']}x{p['instr']}" for p in best_set],
        greedy_set=[f"{p['node']}x{p['instr']}" for p in batch])
    return res


# ----------------------------------------------------------------------------------
# 8. SYMMETRIC-QC: perturb rho +-50% and weights x/÷2; rank stability
# ----------------------------------------------------------------------------------
def spearman(r1, r2):
    common = [k for k in r1 if k in r2]
    n = len(common)
    if n < 2:
        return float("nan")
    d2 = sum((r1[k] - r2[k]) ** 2 for k in common)
    return 1 - 6 * d2 / (n * (n * n - 1))


def rank_map(ranking):
    return {name: i for i, (name, _) in enumerate(ranking)}


def stability_analysis(seed=11, n_trials=40):
    rng = random.Random(seed)
    cands = build_candidates(gpu_cost_mult=SCENARIOS["gpu_rich"]["gpu_cost_mult"])
    base_rank = rank_map(full_ranking(cands, RHO_CAT))
    base_batch, _, _, _ = greedy_select(cands, SCENARIOS["gpu_rich"]["budget"], RHO_CAT)
    base_top = {f"{p['node']}x{p['instr']}" for p in base_batch}

    def one_mode(perturb_rho, perturb_w):
        sps, jacs = [], []
        for _ in range(n_trials):
            rc = {k: (v * rng.uniform(0.5, 1.5) if perturb_rho else v)
                  for k, v in RHO_CAT.items()}
            wt = {n["id"]: n["weight"] * (2 ** rng.uniform(-1, 1) if perturb_w else 1)
                  for n in NODES}
            r = rank_map(full_ranking(cands, rc, wt))
            sps.append(spearman(base_rank, r))
            b, _, _, _ = greedy_select(cands, SCENARIOS["gpu_rich"]["budget"], rc,
                                       weights=wt)
            top = {f"{p['node']}x{p['instr']}" for p in b}
            jacs.append(len(base_top & top) / max(1, len(base_top | top)))
        sps.sort(); jacs.sort()
        mid = len(sps) // 2
        return dict(median_spearman=sps[mid], min_spearman=sps[0],
                    median_top_jaccard=jacs[mid], min_top_jaccard=jacs[0])

    return dict(
        rho_only=one_mode(True, False),
        weights_only=one_mode(False, True),
        both=one_mode(True, True),
        n_trials=n_trials,
        base_top_batch=sorted(base_top),
    )


# ----------------------------------------------------------------------------------
# 9. MAIN — demonstration on the real graph + pre-registered anchors
# ----------------------------------------------------------------------------------
def main():
    evidence = {"script": os.path.basename(__file__),                 "model_params": dict(P0=P0, P_PRIV=P_PRIV, RHO_CAT=RHO_CAT,
                                     PRIOR_BY_KIND=PRIOR_BY_KIND,
                                     INSTRUMENT_COST=INSTRUMENT_COST)}

    # node table (the transcribed instance)
    evidence["nodes"] = [
        dict(id=n["id"], kind=n["kind"], weight=n["weight"], prior=n["prior"],
             n_evidence=len(n["ev"]), cited_by=CITED_BY[n["id"]],
             fa_current=round(node_fa(n, [], RHO_CAT), 6),
             undetected_mass=round(n["weight"] * n["prior"] * node_fa(n, [], RHO_CAT), 6),
             desc=n["desc"])
        for n in NODES]
    total_mass = sum(x["undetected_mass"] for x in evidence["nodes"])
    evidence["total_undetected_failure_mass"] = round(total_mass, 6)

    # law verification
    evidence["floor_law_verification"] = verify_floor_law()

    # scenarios
    evidence["scenarios"] = {}
    for name, sc in SCENARIOS.items():
        cands = build_candidates(exclude=sc["exclude"],
                                 gpu_cost_mult=sc["gpu_cost_mult"])
        batch, trace, g_total, spent = greedy_select(cands, sc["budget"], RHO_CAT)
        ranking = full_ranking(cands, RHO_CAT)
        evidence["scenarios"][name] = dict(
            budget=sc["budget"], excluded=sorted(sc["exclude"]),
            top10_batch=trace,
            batch_total_gain=round(g_total, 6),
            batch_cost=spent,
            fraction_of_mass_resolved=round(g_total / total_mass, 4),
            distinct_instrument_classes_in_batch=sorted({p["instr"] for p in batch}),
            full_ranking_top20=[(nm, round(g, 6)) for nm, g in ranking[:20]],
        )

    # submodularity + greedy gap
    evidence["submodularity"] = submodularity_and_greedy_gap()

    # stability QC
    evidence["stability"] = stability_analysis()

    # -------------------- PRE-REGISTERED ANCHORS --------------------
    anchors = {}
    grich = build_candidates()
    rank_full = full_ranking(grich, RHO_CAT)
    rmap = rank_map(rank_full)
    gains = dict(rank_full)
    key_r4 = "R4_CONTxgpu_microbench"
    key_r2rep = "R2_DETxgpu_microbench"
    all_top10 = {sc: {t["node"] + "x" + t["instr"]
                      for t in evidence["scenarios"][sc]["top10_batch"]}
                 for sc in SCENARIOS}
    a1 = (rmap[key_r4] < rmap[key_r2rep]) and all(
        key_r2rep not in tops for tops in all_top10.values())
    anchors["A1_decorrelated_R4_beats_R2_repeat"] = dict(
        passed=bool(a1),
        rank_R4_gpu=rmap[key_r4], gain_R4_gpu=round(gains[key_r4], 6),
        rank_R2_repeat=rmap[key_r2rep], gain_R2_repeat=round(gains[key_r2rep], 8),
        r2_repeat_in_any_top10=any(key_r2rep in t for t in all_top10.values()))

    top10_gains = sorted((t["marginal_gain"]
                          for t in evidence["scenarios"]["gpu_rich"]["top10_batch"]),
                         reverse=True)
    med_top = top10_gains[len(top10_gains) // 2]
    a2_ratio = gains[key_r2rep] / med_top
    anchors["A2_rho_floor_binds_quantitatively"] = dict(
        passed=bool(a2_ratio < 0.05),
        r2_5th_repeat_gain=round(gains[key_r2rep], 8),
        median_top10_gain=round(med_top, 6),
        ratio=round(a2_ratio, 6))

    bad_picks = []
    for sc, tops in all_top10.items():
        for t in evidence["scenarios"][sc]["top10_batch"]:
            if t["is_repeat"] and NODE_BY_ID[t["node"]]["ev"] and \
                    len(NODE_BY_ID[t["node"]]["ev"]) >= 3:
                bad_picks.append((sc, t["node"], t["instr"]))
    anchors["A3_no_repeat_on_settled_nodes_in_top10"] = dict(
        passed=not bad_picks, offending_picks=bad_picks)

    sm = evidence["submodularity"]
    a4 = min(sm["cardinality_k4"]["ratio"], sm["knapsack_b25"]["ratio"])
    anchors["A4_greedy_gap"] = dict(
        passed=bool(a4 >= 0.95), min_ratio=round(a4, 5),
        dr_violations=sm["violations"], dr_checks=sm["diminishing_returns_checks"])

    st = evidence["stability"]
    a5 = (st["both"]["median_spearman"] >= 0.8 and
          st["both"]["median_top_jaccard"] >= 0.6)
    dominating = ("rho" if st["rho_only"]["median_spearman"] <
                  st["weights_only"]["median_spearman"] else "weights")
    anchors["A5_stability"] = dict(
        passed=bool(a5),
        median_spearman_both=round(st["both"]["median_spearman"], 4),
        median_jaccard_both=round(st["both"]["median_top_jaccard"], 4),
        dominating_input=dominating)

    evidence["preregistered_anchors"] = anchors
    evidence["verdict"] = dict(
        all_anchors_passed=all(a["passed"] for a in anchors.values()),
        anchor_summary={k: a["passed"] for k, a in anchors.items()})
    evidence["declared_limits"] = [
        "objective factorizes per node: cross-node common causes (same local GPU "
        "poisoning many nodes) are outside the objective; batch instrument diversity "
        "reported as mitigation surface, joint N-way co-miss NOT certified "
        "(pairwise-decorrelation != joint coverage, inherited law)",
        "priors PRIOR_BY_KIND are judgment inputs (not perturbed in A5; rho and "
        "weights are). A future rung: perturb priors too.",
        "rho_cat values are engineering estimates, not measured miss-correlations; "
        "the stability analysis prices exactly this uncertainty (+-50%)",
        "submodularity is MEASURED on this instance (A4), not proven for the "
        "posterior-Bernoulli/common-cause gain in general",
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(evidence, f, indent=1)

    # ---- console report ----
    print("=" * 78)
    print("OED PROBE-SELECTION ENGINE — MASTER_UNLOCK_GRAPH instance "
          f"({len(NODES)} nodes, total undetected-failure mass {total_mass:.3f})")
    print("=" * 78)
    flv = evidence["floor_law_verification"]
    ti = flv["two_sensor_identity"]
    print(f"[law] 2-sensor identity err={ti['identity_abs_err']:.2e} "
          f"(rho_ind={ti['rho_ind']:.3f}); repeat-floor q="
          f"{flv['repeat_floor']['common_cause_floor_q']:.4f}: "
          f"FA(k repeats)= " + ", ".join(
              f"k{k}:{v:.4f}" for k, v in flv["repeat_floor"]["fa_by_repeats"].items()))
    print(f"[law] 4 repeats + 1 DECORRELATED probe -> FA="
          f"{flv['repeat_floor']['fa_4_repeats_plus_one_decorrelated']:.4f}  "
          "(the decorrelation dividend the floor law predicts)")
    for name in SCENARIOS:
        sc = evidence["scenarios"][name]
        print(f"\n--- scenario {name} (budget {sc['budget']}, "
              f"excl {sc['excluded'] or 'none'}) — batch gain {sc['batch_total_gain']} "
              f"= {100*sc['fraction_of_mass_resolved']:.1f}% of mass, "
              f"cost {sc['batch_cost']} ---")
        for i, t in enumerate(sc["top10_batch"]):
            print(f"  {i+1:2d}. {t['node']:<16s} x {t['instr']:<18s} "
                  f"gain={t['marginal_gain']:.4f} cost={t['cost']} "
                  f"g/c={t['gain_per_cost']:.4f}"
                  + ("  [REPEAT]" if t["is_repeat"] else ""))
    print("\n--- submodularity / greedy gap ---")
    print(f"  diminishing-returns violations: {sm['violations']}/{sm['dr_checks'] if 'dr_checks' in sm else sm['diminishing_returns_checks']}"
          f" (worst {sm['worst_violation']:.2e})")
    print(f"  cardinality k=4: greedy/opt = {sm['cardinality_k4']['ratio']:.5f}")
    print(f"  knapsack b=25 : greedy/opt = {sm['knapsack_b25']['ratio']:.5f}")
    print("\n--- stability (SYMMETRIC-QC) ---")
    for mode in ("rho_only", "weights_only", "both"):
        s = st[mode]
        print(f"  {mode:<13s} spearman med={s['median_spearman']:.3f} "
              f"min={s['min_spearman']:.3f}  top-batch jaccard "
              f"med={s['median_top_jaccard']:.3f} min={s['min_top_jaccard']:.3f}")
    print("\n--- PRE-REGISTERED ANCHORS ---")
    for k, a in anchors.items():
        print(f"  {'PASS' if a['passed'] else 'FAIL'}  {k}")
    print(f"\nverdict: all_anchors_passed = {evidence['verdict']['all_anchors_passed']}")
    print(f"evidence -> {OUT}")


if __name__ == "__main__":
    main()
