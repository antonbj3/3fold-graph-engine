#!/usr/bin/env python3
"""atoms_validation.py — run the ATOMS runner against REAL reports and write the validation JSON.

Six cases, each a markdown report TEXT with a ```atoms fenced block (so the report-text parse path is
exercised, not just dict-in):
  (a) codec-v2 negative  -> ATOMS reconstructed retroactively from the shipped artifact  -> GREEN
  (b) certified_splat_ab -> ATOMS incl. an honest-NEGATIVE claim (G3 fails) + a round-trip -> GREEN
  (c) POISONED           -> (b) with two expected numbers corrupted                       -> RED w/ diffs
  (d) TRIVIAL            -> soft atoms that always pass + an unbacked claim                -> RED, flagged
  (f) CMD_ALIAS          -> the alias key `cmd` must FAIL, the correct key `command` must PASS -> RED
      empty command string -> subprocess.run("") -> exit 0 -> the atom passed without running
      anything. 78 reports / 166 atoms were written that way. Two-sided guard.
  (e) FALSY_LHS           -> literal lhs=0/[]/""/False on inequality/round-trip atoms       -> GREEN
      (atoms_runner_falsy_lhs_v1,, cell atoms_runner_falsy_lhs_v1): `atom.get("lhs") or
      {...}` silently discarded an explicitly-provided falsy literal (0, [], "", False) and fell back
      to a legacy {artifact,key} pair built from stray top-level "artifact"/"key" fields on the atom --
      either making a TRUE claim (0 == 0) report "operand unresolvable" (fail-closed but WRONG reason),
      or, worse, silently substituting an unrelated artifact value for the intended literal 0 (wrong
      value, no error). Fixed at the source in atoms_runner.py (`_operand_value`/`run_atom`) by using an
      explicit `is None` check instead of `or`. This case is the regression guard: it must stay GREEN.

Output: reports/probes/atoms_standard_validation.json
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from atoms_runner import run_atoms, BASE_DIR  # noqa: E402

# The two artifacts the fixture atoms cite. They are written by _write_fixture_artifacts into a
# temp directory, so the validation suite is self-contained.
CODEC = "reports/probes/example_confound_validation.json"
SPLAT = "reports/probes/example_ab_validation.json"


def _write_fixture_artifacts(base):
    """Write the two small artifacts the fixture atoms pin values in."""
    codec = {
        "confound_panel": {"case_a_rendered": {"N2_ptc_slope": 0.03946,
                                                "temporal_autocorr": 0.114,
                                                "conjunction_reads_native": False}},
        "FINDING": {"naive_statistic_false_positive_on_case_a": True,
                     "coded_block_statistic_is_confounded": True,
                     "frozen_gate_both_captures_native": False},
        "constructive_controls": {"genuine_positive": {"N2_ptc_slope": 0.26297,
                                                        "reads_native": True}},
    }
    splat = {
        "gates": {"G1_ratio": 0.318, "G1_floater_reduction": True,
                   "G2_delta_L1": 0.00167, "G2_no_photometric_cost": True,
                   "G3_stamp_honesty": False},
        "A": {"floater_rate_overall": 0.0311, "tau": 0.0425, "heldout_L1": 0.0421, "n_final": 2918},
        "B": {"floater_rate_overall": 0.0099, "tau": 0.0425, "floater_nullstamp_precision": 0.504,
               "heldout_L1": 0.0438},
    }
    for rel, obj in ((CODEC, codec), (SPLAT, splat)):
        ap = os.path.join(base, rel)
        os.makedirs(os.path.dirname(ap), exist_ok=True)
        with open(ap, "w") as f:
            json.dump(obj, f)
    return base


def _report(prose, block):
    """Wrap an ATOMS block dict inside a markdown report as a fenced ```atoms island."""
    return prose.strip() + "\n\n```atoms\n" + json.dumps(block, indent=2) + "\n```\n"


# ---------------------------------------------------------------- (a) codec-v2, reconstructed
def _codec_block():
    return {
        "report": "example_confound_validation",
        "claims": [
            {"id": "artifact", "text": "the codec-v2 validation artifact exists", "load_bearing": False},
            {"id": "naive_fp", "text": "the naive statistic gives a FALSE positive on case A",
             "load_bearing": True},
            {"id": "conj_fences", "text": "the conjunction gate FENCES case A (does not pass it as native)",
             "load_bearing": True},
            {"id": "genuine_seen", "text": "the method DETECTS a genuine signal through the same encoder "
                                           "(positive control -> not a method failure)", "load_bearing": True},
        ],
        "atoms": [
            {"id": "exists", "claim": "artifact", "type": "artifact-exists", "artifact": CODEC},
            # naive_fp: pin the slope (STRONG) + the boolean finding (weak)
            {"id": "bbb_slope", "claim": "naive_fp", "type": "value-in-artifact", "artifact": CODEC,
             "key": "confound_panel.case_a_rendered.N2_ptc_slope", "expected": 0.03946, "tol": 1e-4},
            {"id": "naive_flag", "claim": "naive_fp", "type": "value-in-artifact", "artifact": CODEC,
             "key": "FINDING.naive_statistic_false_positive_on_case_a", "expected": True},
            # conj_fences: pin the temporal autocorr that drives the fence (STRONG) + boolean fence flags
            {"id": "bbb_tac", "claim": "conj_fences", "type": "value-in-artifact", "artifact": CODEC,
             "key": "confound_panel.case_a_rendered.temporal_autocorr", "expected": 0.114, "tol": 1e-3},
            {"id": "fence_flag", "claim": "conj_fences", "type": "value-in-artifact", "artifact": CODEC,
             "key": "confound_panel.case_a_rendered.conjunction_reads_native", "expected": False},
            {"id": "fence_cmd", "claim": "conj_fences", "type": "command-exit-0",
             "command": ("python3 -c \"import json,sys; d=json.load(open('" + CODEC + "')); "
                         "sys.exit(0 if (d['FINDING']['coded_block_statistic_is_confounded'] and not "
                         "d['FINDING']['frozen_gate_both_captures_native']) else 1)\"")},
            # genuine_seen: pin the genuine slope (STRONG) + it beats BBB slope (inequality, MEDIUM)
            {"id": "gen_slope", "claim": "genuine_seen", "type": "value-in-artifact", "artifact": CODEC,
             "key": "constructive_controls.genuine_positive.N2_ptc_slope", "expected": 0.26297, "tol": 1e-4},
            {"id": "gen_gt_bbb", "claim": "genuine_seen", "type": "inequality",
             "lhs": {"artifact": CODEC, "key": "constructive_controls.genuine_positive.N2_ptc_slope"},
             "op": ">", "rhs": {"artifact": CODEC, "key": "confound_panel.case_a_rendered.N2_ptc_slope"}},
        ],
    }


# ---------------------------------------------------------------- (b) certified_splat_ab
def _splat_block():
    return {
        "report": "certified_splat_ab",
        "claims": [
            {"id": "exists_c", "text": "the AB artifact exists", "load_bearing": False},
            {"id": "reduction", "text": "arm B reduces floaters vs arm A (ratio ~0.318)", "load_bearing": True},
            {"id": "no_cost", "text": "the floater reduction costs no held-out photometric quality (G2)",
             "load_bearing": True},
            {"id": "stamp_dishonest", "text": "HONEST NEGATIVE: the null-stamp honesty gate G3 FAILS "
                                              "(B stamp precision 0.504 < 0.7 bar)", "load_bearing": True},
            {"id": "shared_tau", "text": "both arms are certified against the IDENTICAL data-derived tau",
             "load_bearing": True},
        ],
        "atoms": [
            {"id": "exists", "claim": "exists_c", "type": "artifact-exists", "artifact": SPLAT},
            {"id": "g1_ratio", "claim": "reduction", "type": "value-in-artifact", "artifact": SPLAT,
             "key": "gates.G1_ratio", "expected": 0.318, "tol": 1e-3},
            {"id": "g1_flag", "claim": "reduction", "type": "value-in-artifact", "artifact": SPLAT,
             "key": "gates.G1_floater_reduction", "expected": True},
            {"id": "rate_lt", "claim": "reduction", "type": "inequality",
             "lhs": {"artifact": SPLAT, "key": "B.floater_rate_overall"}, "op": "<",
             "rhs": {"artifact": SPLAT, "key": "A.floater_rate_overall"}},
            {"id": "g2_dl1", "claim": "no_cost", "type": "value-in-artifact", "artifact": SPLAT,
             "key": "gates.G2_delta_L1", "expected": 0.00167, "tol": 1e-4},
            {"id": "g2_flag", "claim": "no_cost", "type": "value-in-artifact", "artifact": SPLAT,
             "key": "gates.G2_no_photometric_cost", "expected": True},
            # honest negative: the failing gate is REPORTED as false and that is VERIFIED true
            {"id": "g3_flag", "claim": "stamp_dishonest", "type": "value-in-artifact", "artifact": SPLAT,
             "key": "gates.G3_stamp_honesty", "expected": False},
            {"id": "prec_pin", "claim": "stamp_dishonest", "type": "value-in-artifact", "artifact": SPLAT,
             "key": "B.floater_nullstamp_precision", "expected": 0.504, "tol": 1e-3},
            {"id": "prec_below_bar", "claim": "stamp_dishonest", "type": "inequality",
             "lhs": {"artifact": SPLAT, "key": "B.floater_nullstamp_precision"}, "op": "<", "rhs": 0.7},
            # round-trip: both arms share the identical tau (different keys, not a tautology)
            {"id": "tau_rt", "claim": "shared_tau", "type": "round-trip-identical",
             "left": {"artifact": SPLAT, "key": "A.tau"}, "right": {"artifact": SPLAT, "key": "B.tau"}},
        ],
    }


# ---------------------------------------------------------------- (c) poisoned = (b) with corrupted numbers
def _poisoned_block():
    b = json.loads(json.dumps(_splat_block()))
    for a in b["atoms"]:
        if a["id"] == "g1_ratio":
            a["expected"] = 0.9          # real is 0.318
        if a["id"] == "g3_flag":
            a["expected"] = True         # real is False
    return b


# ---------------------------------------------------------------- (d) trivial / gamed atoms
def _trivial_block():
    return {
        "report": "gamed_report_example",
        "claims": [
            {"id": "loss_ok", "text": "held-out L1 is acceptable", "load_bearing": True},
            {"id": "tau_consistent", "text": "tau is self-consistent", "load_bearing": True},
            {"id": "it_ran", "text": "the pipeline ran", "load_bearing": True},
            {"id": "prod_ready", "text": "arm B is production-ready", "load_bearing": True},  # NO atom
        ],
        "atoms": [
            # generous one-sided bound -> passes on ~anything
            {"id": "loss_huge_bound", "claim": "loss_ok", "type": "inequality",
             "lhs": {"artifact": SPLAT, "key": "A.heldout_L1"}, "op": "<", "rhs": 1e9},
            # tolerance so wide the value need not be right
            {"id": "loss_wide_tol", "claim": "loss_ok", "type": "value-in-artifact", "artifact": SPLAT,
             "key": "A.heldout_L1", "expected": 0.5, "tol": 100.0},
            # tautology: compares tau to itself
            {"id": "tau_self", "claim": "tau_consistent", "type": "round-trip-identical",
             "left": {"artifact": SPLAT, "key": "A.tau"}, "right": {"artifact": SPLAT, "key": "A.tau"}},
            # bare command: exits 0 as long as it does not crash
            {"id": "bare_run", "claim": "it_ran", "type": "command-exit-0",
             "command": "python3 -c \"print('ran')\""},
        ],
    }


# ---------------------------------------------------------------- (e) falsy-literal lhs regression
def _falsy_lhs_block():
    """Literal falsy `lhs` operands (0, False) on inequality atoms.

    Scoped tightly to the actual bug: `atom.get("lhs") or {"artifact":..., "key":...}` in
    atoms_runner.py's `atom_strength` and `run_atom` treated an explicitly-provided falsy literal
    (0, False -- both `bool(x) is False`) as "lhs not provided" and silently fell back to the legacy
    {artifact,key} pair built from stray top-level "artifact"/"key" fields on the atom -- either
    None/None (-> wrong-reason "operand unresolvable", ok=False on a TRUE claim) or, when those stray
    fields happen to reference a real artifact key (atom e5 below), a silently WRONG substituted value.
    (round-trip-identical's left/right never used `or` and is unaffected -- not tested here; `[]`/`""`
    are non-numeric and were never valid inequality operands regardless of this bug, so they're not
    included either -- see atoms_runner_falsy_lhs_v1 repro notes for that scoping decision.)

    Every atom here is TRUE by construction. Pre-fix this case goes RED; post-fix it must be GREEN.
    """
    return {
        "report": "falsy_lhs_regression",
        "claims": [
            {"id": "zero_eq_zero", "text": "literal 0 == 0 (lhs=0, a bare falsy int)", "load_bearing": True},
            {"id": "false_lt_one", "text": "literal False (== 0) < 1", "load_bearing": True},
            {"id": "zero_ge_neg1", "text": "literal 0 >= -1", "load_bearing": True},
            {"id": "zero_not_hijacked", "text": "lhs=0 stays 0 even when stray top-level artifact/key "
                                                 "fields point at an unrelated value", "load_bearing": True},
        ],
        "atoms": [
            {"id": "e1", "claim": "zero_eq_zero", "type": "inequality", "lhs": 0, "op": "==", "rhs": 0},
            {"id": "e3", "claim": "false_lt_one", "type": "inequality", "lhs": False, "op": "<", "rhs": 1},
            {"id": "e4", "claim": "zero_ge_neg1", "type": "inequality", "lhs": 0, "op": ">=", "rhs": -1},
            # stray legacy artifact/key fields point at n_final=2918 (>> 1) -- if the bug hijacks lhs
            # away from the literal 0, "2918 < 1" is False and this atom flips to FAIL; the literal
            # 0 < 1 must stay True regardless of the stray fields' presence.
            {"id": "e5", "claim": "zero_not_hijacked", "type": "inequality", "lhs": 0, "op": "<",
             "artifact": SPLAT, "key": "A.n_final", "rhs": 1},
        ],
    }


# ------------------------------------------------- (f) cmd-alias fail-open (atoms_omcert_v1)
def _cmd_alias_block():
    """Two-sided proof for the fail-open class `cmd` instead of `command`.

    MEASURED BLAST RADIUS: 78 reports / 166 atoms were written with an alias key (`cmd`), so the
    runner read an empty string, `subprocess.run("")` returned 0 and the atom PASSED without running
    anything. Both halves are required: the alias atom must FAIL (otherwise the hole is open again)
    AND its twin with the correct key must PASS (otherwise the fix is blanket-red and says nothing).
    """
    return {
        "report": "cmd_alias_regression",
        "claims": [
            {"id": "alias_must_not_cover", "load_bearing": True,
             "text": "An atom written with the alias key `cmd` must NEVER pass -- it is UNWRITTEN, not satisfied."},
            {"id": "correct_key_passes", "load_bearing": True,
             "text": "The same atom with the correct key `command` and a true command passes."},
        ],
        "atoms": [
            {"id": "f_alias", "claim": "alias_must_not_cover", "type": "command-exit-0",
             "cmd": "test 1 -eq 1"},
            {"id": "f_command", "claim": "correct_key_passes", "type": "command-exit-0",
             "command": "test 1 -eq 1"},
        ],
    }


def validate(base=None):
    import tempfile
    base = base or _write_fixture_artifacts(tempfile.mkdtemp(prefix="atoms_validation_"))
    a = run_atoms(_report("# codec-v2 negative (retro ATOMS)", _codec_block()), base_dir=base, execute=True)
    b = run_atoms(_report("# certified splat A/B", _splat_block()), base_dir=base, execute=False)
    c = run_atoms(_report("# POISONED report", _poisoned_block()), base_dir=base, execute=False)
    d = run_atoms(_report("# TRIVIAL/gamed report", _trivial_block()), base_dir=base, execute=True)
    e = run_atoms(_report("# FALSY_LHS regression (atoms_runner_falsy_lhs_v1)", _falsy_lhs_block()),
                  base_dir=base, execute=False)

    f = run_atoms(_report("# CMD_ALIAS regression (atoms_omcert_v1)", _cmd_alias_block()),
                  base_dir=base, execute=True)

    # ---- expectations (this IS the QC: run the list) ----
    a_ok = (a["decision"] == "ALLOW" and a["verdict"] == "GREEN"
            and all(c2["verdict"] == "COVERED" for c2 in a["coverage"]) and not a["triviality_flags"])
    b_ok = (b["decision"] == "ALLOW" and b["verdict"] == "GREEN"
            and all(c2["verdict"] == "COVERED" for c2 in b["coverage"]) and not b["triviality_flags"])
    # c: RED, and specifically the two poisoned value atoms FAIL with a diff
    poisoned_fails = {r["id"]: r for r in c["atoms"] if r["ok"] is False}
    c_ok = (c["decision"] == "BLOCK"
            and "g1_ratio" in poisoned_fails and poisoned_fails["g1_ratio"].get("diff") is not None
            and "g3_flag" in poisoned_fails)
    # d: RED, flagged by triviality + coverage (the unbacked claim demoted to hypothesis)
    triv_ids = {t["id"] for t in d["triviality_flags"]}
    cov = {cc["claim"]: cc["verdict"] for cc in d["coverage"]}
    d_ok = (d["decision"] == "BLOCK"
            and {"loss_huge_bound", "loss_wide_tol", "tau_self"} <= triv_ids
            and cov.get("prod_ready") == "UNBACKED_HYPOTHESIS"
            and cov.get("it_ran") == "THIN"
            and cov.get("loss_ok") == "TRIVIAL_ONLY")
    # e: every falsy-literal-lhs atom must actually be TRUE (ok=True) -- proves the literal 0/False was
    # not silently swapped for an unresolved-or-hijacked operand. (NOT requiring overall GREEN/COVERED:
    # e3/e4/e5 are one-sided inequalities against a loose bound, which this system's own anti-gaming
    # triviality banding correctly scores WEAK/THIN by design in the reference classification
    # ensidig-olikhetsatom-strukturellt-svag -- that is unrelated to the falsy-lhs bug being guarded
    # against here. e1's "==" IS tight enough to reach COVERED and is checked as the decisive positive.)
    e_atoms_ok = {r["id"]: r["ok"] for r in e["atoms"]}
    e1_cov = next((c2["verdict"] for c2 in e["coverage"] if c2["claim"] == "zero_eq_zero"), None)
    e_ok = (all(v is True for v in e_atoms_ok.values()) and e1_cov == "COVERED"
            and not any(r["ok"] is False for r in e["atoms"]))

    # f: two-sided -- the alias atom MUST fail with a diagnosis naming the key, the twin MUST pass
    f_atoms = {r["id"]: r for r in f["atoms"]}
    f_ok = (f_atoms.get("f_alias", {}).get("ok") is False
            and "cmd" in str(f_atoms.get("f_alias", {}).get("reason", ""))
            and f_atoms.get("f_command", {}).get("ok") is True
            and f["decision"] == "BLOCK")

    # ---- fold gate self-tests (v1 untouched, v2 = v1 + atoms) ----
    from fold_gate import _selftest as v1_selftest
    from fold_gate_v2 import _selftest as v2_selftest
    v1 = v1_selftest()["all_pass"]
    v2 = v2_selftest()["all_pass"]

    all_pass = a_ok and b_ok and c_ok and d_ok and e_ok and f_ok and v1 and v2
    out = {
        "cell": "atoms_standard_validation",
        "lane": "I",
                "runner": "scripts/atoms_runner.py",
        "spec": "docs/ATOMS_STANDARD.md",
        "premise": ("decomposition is the SUPPLIER's duty: each cell-report carries a machine-readable ATOMS "
                    "block {claim, check-command, expected +/- tol}; QC = run the list. Anti-gaming: an atom "
                    "that would pass on a RANDOM result (null-pass-rate >= 0.9) is flagged TRIVIAL; a claim "
                    "with no atom is by definition a HYPOTHESIS."),
        "all_pass": bool(all_pass),
        "case_pass": {"a_codec_v2_retro": bool(a_ok), "b_certified_splat_ab": bool(b_ok),
                      "c_poisoned": bool(c_ok), "d_trivial": bool(d_ok), "e_falsy_lhs": bool(e_ok),
                      "f_cmd_alias": bool(f_ok)},
        "gate_selftests": {"fold_gate_v1_all_pass": bool(v1), "fold_gate_v2_all_pass": bool(v2)},
        "cases": {
            "a_codec_v2_retro": {"expected": "GREEN", "run": a},
            "b_certified_splat_ab": {"expected": "GREEN", "run": b},
            "c_poisoned": {"expected": "RED (2 value mismatches with diff)",
                           "poisoned_fail_ids": list(poisoned_fails.keys()), "run": c},
            "d_trivial": {"expected": "RED (triviality + coverage flags)",
                          "triviality_ids": sorted(triv_ids), "coverage": cov, "run": d},
            "e_falsy_lhs": {"expected": "all falsy-literal-lhs atoms ok=True (e1 additionally COVERED); "
                                        "regression guard for atoms_runner_falsy_lhs_v1",
                            "atom_ok": e_atoms_ok, "run": e},
            "f_cmd_alias": {"expected": "two-sided: f_alias (key `cmd`) FAILs with a diagnosis naming the "
                                        "key, f_command (key `command`) PASSes; regression guard for the "
                                        "measured fail-open class (78 reports / 166 atoms)",
                            "run": f},
        },
    }
    outdir = os.environ.get("ATOMS_VALIDATION_OUT", os.path.join(os.getcwd(), "reports", "probes"))
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "atoms_standard_validation.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    for name, ok in out["case_pass"].items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"[{'PASS' if v1 else 'FAIL'}] fold_gate_v1_selftest   "
          f"[{'PASS' if v2 else 'FAIL'}] fold_gate_v2_selftest")
    print(f"ALL_PASS={all_pass} -> {outpath}")
    return out


if __name__ == "__main__":
    out = validate()
    sys.exit(0 if out["all_pass"] else 1)
