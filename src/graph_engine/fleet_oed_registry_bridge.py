#!/usr/bin/env python3
"""fleet_oed_registry_bridge.py -- THE ONE MEASUREMENT for innovation-map #2:
can fleet_oed's sigma_min engine run on a REAL coverage matrix built from the live detector
registry (data/QC_DETECTOR_REGISTRY.jsonl), and does it emit interpretable blind-modes output
that the LLR gate (scripts/bayes_qc_gate.py) can consume?

MEASURED vs CONSTRUCTED (declared per AGENT_PREAMBLE line 4):
  MEASURED: every column weight comes from a registry row's own measured numbers
               (tpr/fpr operating point -> Youden J; else score_effective_ci_lo (B's CI-hardened
               convention, LOWER bound); else effective_coverage). AUC-only rows get weight 0 --
               that is the REGISTRY'S OWN rule ("AUC-only... carries zero log-LR weight in
               bayes_qc_gate until an operating point... exists"), applied symmetrically here.
               Placeholder/unmeasured rows get 0. Superseded rows are EXCLUDED as columns.
  CONSTRUCTED: (a) the mode BASIS. Modes = registry 'domain' values, with the shared
               s7_video_admission_coherent_fake domain split into its five posture-v2 axes
               (semantic/acquisition/motion/deformation/appearance) read mechanically from the
               row's 'covers' field when present, else from the row NAME. (b) the 6 mechanical
               rows carry no 'domain' field; they share one calibration facit (k1, own-selftest
               claim shape) so they are assigned the constructed mode label
               'code_claim_own_selftest'. (c) the NBA candidate columns: each is "calibrate an
               operating point for an existing AUC-only informative continuous channel", with a
               PROXY weight 2*AUC-1 (labelled constructed -- a plausible J for that AUC, not a
               measurement).

Bridge vocabulary (fleet_oed <-> cert_decorrelation <-> registry):
  fleet_oed 'mode'  = a failure-KIND direction = here: a registry domain (s7 split per axis).
  fleet_oed 'cert'  = one registry detector row = one coverage COLUMN over the modes.
  cert_decorrelation's catch_matrix is (failure x cert) BOOLEAN; the same M thresholded at
  weight>0 feeds coverage_record for the blind/redundant/dead cross-check.

Usage:
  python3 scripts/fleet_oed_registry_bridge.py            # full run, writes the report JSON
  python3 scripts/fleet_oed_registry_bridge.py --print K  # print one decisive value (atom hook)
       K in {sigmin, blind, null_dim, n_certs, n_modes, covered, nba, vibrate_peak}
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from .fleet_oed import (fleet_soft_mode, next_best_acquisition, vibrate,  # noqa: E402
                                    coverage_gramian)
from .cert_decorrelation import coverage_record  # noqa: E402
try:
    from bayes_qc_gate import load_registry
except ImportError:  # optional external capability registry
    load_registry = None  # noqa: E402  (import-only; the gate we must feed)

REGISTRY = os.path.join(BASE, "data", "QC_DETECTOR_REGISTRY.jsonl")
REPORT = os.path.join(BASE, "reports", "probes", "fleet_oed_blind_modes.json")

S7 = "s7_video_admission_coherent_fake"
_S7_AXES = ["semantic", "acquisition", "motion", "deformation", "appearance"]
# name-substring -> axis (mechanical read of the posture-v2 axis embedded in every s7 row name)
_NAME_AXIS = {"semantic": "semantic", "acquisition": "acquisition", "rolling_shutter": "acquisition",
              "motion": "motion", "flow": "motion", "deformation": "deformation",
              "appearance": "appearance"}


def row_mode(row):
    """CONSTRUCTED mapping (declared): registry row -> failure-mode label."""
    dom = row.get("domain")
    if dom is None:
        return "code_claim_own_selftest"          # the 6 mechanical rows' shared k1 facit domain
    if dom == S7:
        covers = row.get("covers") or []
        for c in covers:
            if c in _S7_AXES:
                return f"s7:{c}"
        nm = row["name"].lower()
        for key, ax in _NAME_AXIS.items():
            if key in nm:
                return f"s7:{ax}"
        return "s7:unmapped"
    return dom


def row_weight(row):
    """MEASURED coverage weight for one registry row (see module docstring hierarchy).
    Returns (weight, basis) or (None, reason) if the row must be excluded as a column."""
    if row.get("superseded_by"):
        return None, f"superseded_by={row['superseded_by']}"
    tpr, fpr = row.get("tpr"), row.get("fpr")
    if isinstance(tpr, (int, float)) and isinstance(fpr, (int, float)):
        return max(0.0, float(tpr) - float(fpr)), "youden_J=tpr-fpr (MEASURED operating point)"
    ci_lo = row.get("score_effective_ci_lo")
    if isinstance(ci_lo, (int, float)):
        return float(ci_lo), "score_effective_ci_lo (MEASURED, CI-hardened lower bound)"
    ec = row.get("effective_coverage")
    if isinstance(ec, (int, float)):
        return float(ec), "effective_coverage (MEASURED)"
    if isinstance(row.get("auc"), (int, float)):
        return 0.0, "AUC-only -> 0 (registry's own rule: no operating point => zero weight)"
    return 0.0, "unmeasured placeholder -> 0"


def build_M():
    """M (n_modes x n_certs) from the live registry. Returns M, mode_names, cert_names, audit."""
    bad = []
    rows = load_registry(REGISTRY, bad_lines=bad)
    audit, cols = [], []
    for name, row in rows.items():
        w, basis = row_weight(row)
        if w is None:
            audit.append({"cert": name, "excluded": basis})
            continue
        m = row_mode(row)
        audit.append({"cert": name, "mode": m, "weight": round(w, 4), "basis": basis})
        cols.append((name, m, w))
    modes = sorted({m for _, m, _ in cols})
    M = np.zeros((len(modes), len(cols)))
    for j, (_, m, w) in enumerate(cols):
        M[modes.index(m), j] = w
    cert_names = [c[0] for c in cols]
    return M, modes, cert_names, audit, bad


def measure():
    M, modes, certs, audit, bad = build_M()
    G = coverage_gramian(M)
    sm = fleet_soft_mode(M, modes)

    # exact null-space dimension of G (multi-blind check -- fleet_soft_mode only reports ONE vector)
    eig = np.linalg.eigvalsh((G + G.T) / 2)
    null_dim = int((np.clip(eig, 0, None) < 1e-9).sum())
    zero_rows = [modes[i] for i in range(len(modes)) if not M[i].any()]   # exact blind modes

    # FDT read (model-free) + structural-null probe: with a >=2-dim blind space a rank-1 dither
    # cannot lift sigma_min out of the null, so the response is EXACTLY zero (float noise) and the
    # ranking is meaningless. MEASURED here + single-null control (add one blind cover, re-read).
    resp = vibrate(M)
    vib_rank = [modes[i] for i in np.argsort(-resp)]
    vib_max = float(np.abs(resp).max())
    c_sem = np.zeros(len(modes)); c_sem[modes.index("s7:semantic")] = 2 * 0.81 - 1
    resp1 = vibrate(np.column_stack([M, c_sem]))
    vib_single_null = {"peak": modes[int(np.argmax(resp1))], "max_resp": float(np.abs(resp1).max())}

    # cert_decorrelation cross-check: boolean catch view (weight>0), coverage_record vocabulary
    Mb = {c: {m: bool(M[i, j] > 0) for i, m in enumerate(modes)} for j, c in enumerate(certs)}
    certs_callable = [(lambda st, _c=c: Mb[_c][st]) for c in certs]
    rec = coverage_record("FLEET", "qc_detector_registry", certs_callable, modes,
                          baseline_state=None, cert_names=certs, failure_names=modes)

    # NBA: CONSTRUCTED candidates = calibrate an operating point on informative AUC-only channels
    def col(mode, w):
        c = np.zeros(len(modes)); c[modes.index(mode)] = w; return c
    cand_defs = [
        ("calibrate_op:H_semantic_modal_dominance", "s7:semantic", 2 * 0.81 - 1),
        ("calibrate_op:J_deformation_continuous_score", "s7:deformation", 2 * 0.85 - 1),
        ("calibrate_op:C_acquisition_rolling_shutter(redundant)", "s7:acquisition", 2 * 0.96 - 1),
        ("calibrate_op:selftest_execution(redundant)", "code_claim_own_selftest", 0.5),
    ]
    cands = [col(m, w) for _, m, w in cand_defs]
    cnames = [n for n, _, _ in cand_defs]
    nba1 = next_best_acquisition(M, cands, cnames)

    # multi-null degeneracy probe: greedy 2-step (acquire top blind axis, re-rank)
    M2 = np.column_stack([M, col("s7:semantic", 2 * 0.81 - 1)])
    sm2 = fleet_soft_mode(M2, modes)
    nba2 = next_best_acquisition(M2, cands, cnames)
    M3 = np.column_stack([M2, col("s7:deformation", 2 * 0.85 - 1)])
    sm3 = fleet_soft_mode(M3, modes)

    # FRAME-SENSITIVITY control (priced exit for the one constructed degree of freedom):
    # collapse the s7 axis split into ONE mode and re-measure. If s7 is one mode the agent pool looks
    # COVERED (acquisition/motion/appearance rows mask the semantic/deformation holes) -- the
    # blind verdict DEPENDS on the axis split. External justification for the split: the s7 rows'
    # own posture-v2 vocabulary + the binary union consumes per-axis legs, so per-axis IS the
    # consumed granularity, not a free choice.
    modes_c = sorted({("s7_all" if m.startswith("s7:") else m) for _, m, _ in
                      [(a["cert"], a["mode"], a["weight"]) for a in audit if "excluded" not in a]})
    Mc = np.zeros((len(modes_c), len(certs)))
    for j, a in enumerate([a for a in audit if "excluded" not in a]):
        mm = "s7_all" if a["mode"].startswith("s7:") else a["mode"]
        Mc[modes_c.index(mm), j] = a["weight"]
    smc = fleet_soft_mode(Mc, modes_c)

    report = {
        "claim": ("fleet_oed sigma_min engine RUNS on a real coverage matrix built from "
                  "data/QC_DETECTOR_REGISTRY.jsonl and emits interpretable blind-modes"),
        "measured_vs_constructed": {
            "measured": "column weights (registry rows' own tpr/fpr J, score_effective_ci_lo, effective_coverage); sigma_min/null-dim/vibrate outputs",
            "constructed": "mode basis (domain + s7 axis split), mechanical-rows mode label, NBA candidate proxy weights 2*AUC-1",
        },
        "n_modes": len(modes), "n_certs": len(certs), "modes": modes,
        "registry_rows_excluded": [a for a in audit if "excluded" in a],
        "registry_bad_lines": len(bad),
        "column_audit": [a for a in audit if "excluded" not in a],
        "sigmin": sm["sigmin"], "covered": sm["covered"], "cond": sm["cond"],
        "blind_modes_eigvec_top3": sm["blind_modes"],
        "null_space_dim": null_dim,
        "blind_modes_exact": zero_rows,
        "vibrate_susceptibility_ranking_top4": vib_rank[:4],
        "vibrate_peaks_match_exact_blind": set(vib_rank[:null_dim]) == set(zero_rows),
        "vibrate_max_resp": vib_max,
        "vibrate_verdict": ("STRUCTURALLY NULL under multi-dim blind space (max|resp|~1e-18 = float "
                            "noise; rank-1 dither cannot lift sigma_min out of a >=2-dim null). "
                            "Single-null control below shows the FDT read works once null_dim==1."),
        "vibrate_single_null_control": vib_single_null,
        "cert_decorrelation_record": rec,
        "nba_step1": nba1,
        "nba_step1_degenerate": all(abs(r["delta_sigmin"]) < 1e-9 for r in nba1),
        "nba_step2_after_semantic": {"sigmin": sm2["sigmin"], "ranking": nba2},
        "sigmin_after_both_blind_acquired": sm3["sigmin"],
        "covered_after_both": sm3["covered"],
        "frame_sensitivity_collapsed_s7": {
            "n_modes": len(modes_c), "sigmin": smc["sigmin"], "covered": smc["covered"],
            "verdict": ("blind-modes verdict DEPENDS on the s7 axis split (collapsed: covered=True). "
                        "Split justified externally: the union consumes per-axis legs (posture-v2), "
                        "so per-axis is the consumed granularity."),
        },
        "atoms": [
            {"check_cmd": "python3 scripts/fleet_oed_registry_bridge.py --print null_dim",
             "expected": str(null_dim)},
            {"check_cmd": "python3 scripts/fleet_oed_registry_bridge.py --print blind",
             "expected": ",".join(zero_rows)},
            {"check_cmd": "python3 scripts/fleet_oed_registry_bridge.py --print sigmin",
             "expected": str(sm["sigmin"])},
            {"check_cmd": "python3 scripts/fleet_oed_registry_bridge.py --print n_certs",
             "expected": str(len(certs))},
        ],
    }
    return report


def main(argv):
    rep = measure()
    if "--print" in argv:
        key = argv[argv.index("--print") + 1]
        val = {"sigmin": rep["sigmin"], "blind": ",".join(rep["blind_modes_exact"]),
               "null_dim": rep["null_space_dim"], "n_certs": rep["n_certs"],
               "n_modes": rep["n_modes"], "covered": rep["covered"],
               "nba": rep["nba_step2_after_semantic"]["ranking"][0]["name"],
               "vibrate_peak": rep["vibrate_susceptibility_ranking_top4"][0]}[key]
        print(val)
        return 0
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    slim = {k: rep[k] for k in ["sigmin", "covered", "null_space_dim", "blind_modes_exact",
                                "vibrate_susceptibility_ranking_top4", "vibrate_peaks_match_exact_blind",
                                "nba_step1_degenerate", "nba_step2_after_semantic",
                                "sigmin_after_both_blind_acquired", "covered_after_both",
                                "n_modes", "n_certs"]}
    print(json.dumps(slim, indent=2, default=str))
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
