#!/usr/bin/env python3
"""fold_gate_v2.py — v1's harvest gate PLUS the ATOMS coverage/triviality gate.

BUILDS ON (does not replace) scripts/fold_gate.py:
  v1 (fold_gate) verifies the fold-submission carries a re-run receipt whose cited numbers actually match
     the artifact on disk; negatives need mechanism + a signal-vs-method control; HYPOTHESIS folds as
     hypothesis only.
  v2 additionally requires the SAME report to carry an ATOMS block (docs/ATOMS_STANDARD.md) that
     atoms_runner passes: every atom PASS, every load-bearing claim COVERED by >=1 non-weak passing atom,
     and no atom flagged TRIVIAL by the anti-gaming null-pass-rate check.

  decision = ALLOW iff  v1.decision == ALLOW  AND  atoms.decision == ALLOW.
  (A HYPOTHESIS-tagged submission short-circuits to ALLOW-as-hypothesis exactly as in v1 -- a hypothesis
   is not required to carry a full ATOMS receipt; it is not being booked as a result.)

INPUT: one report that is BOTH a fold-submission and an ATOMS carrier. Two supported shapes:
   * a dict/JSON with a 'rerun' receipt (for v1) and an 'ATOMS' key (for atoms_runner), or
   * a dict/JSON fold-submission + a separate `atoms_report` argument (text or dict) holding the block.

Callable:  fold_gate_v2(submission, atoms_report=None, base_dir=None, execute=False, timeout=110) -> dict
CLI:       python3 scripts/fold_gate_v2.py <submission.json>
           python3 scripts/fold_gate_v2.py --execute <submission.json>
           python3 scripts/fold_gate_v2.py --selftest
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from fold_gate import fold_gate, BASE_DIR, _coerce, _HYP        # noqa: E402
from atoms_runner import run_atoms, parse_atoms_block            # noqa: E402


def fold_gate_v2(submission, atoms_report=None, base_dir=None, execute=False, timeout=110):
    base_dir = base_dir or BASE_DIR
    v1 = fold_gate(submission, base_dir=base_dir, execute=execute, timeout=timeout)

    # HYPOTHESIS short-circuit: v1 already ALLOWed-as-hypothesis; do not demand a full ATOMS receipt.
    if v1.get("as") == "hypothesis":
        return {"decision": "ALLOW", "as": "hypothesis", "v1": v1, "atoms": None,
                "missing": [], "note": "HYPOTHESIS: folded as hypothesis; ATOMS receipt not required."}

    # locate the ATOMS block: explicit arg > submission['ATOMS'] > submission itself (fenced text/dict)
    sub, _raw = _coerce(submission)
    src = atoms_report
    if src is None and isinstance(sub, dict):
        src = {"ATOMS": sub["ATOMS"]} if "ATOMS" in sub else sub
    atoms = run_atoms(src if src is not None else submission,
                      base_dir=base_dir, execute=execute, timeout=min(timeout, 30))

    missing = list(v1.get("missing", []))
    if atoms.get("decision") != "ALLOW":
        missing = missing + [f"ATOMS: {m}" for m in atoms.get("missing", [])]

    decision = "ALLOW" if (v1["decision"] == "ALLOW" and atoms.get("decision") == "ALLOW") else "BLOCK"
    return {"decision": decision, "as": "result",
            "verdict": v1.get("verdict"),
            "v1_decision": v1["decision"], "atoms_decision": atoms.get("decision"),
            "missing": missing, "v1": v1, "atoms": atoms}


# ---------------------------------------------------------------- self-test: v1 cases carried into v2
def _selftest():
    """v1 receipt + ATOMS coverage on a synthetic artifact in a temp directory."""
    import tempfile
    from fold_gate import _synthetic_artifact

    tmp = tempfile.mkdtemp(prefix="fold_gate_v2_selftest_")
    art = _synthetic_artifact(tmp)
    cell_rel = "example_cell.py"
    with open(os.path.join(tmp, cell_rel), "w") as f:
        f.write("# example cell: regenerates the artifact cited by the receipt\n")

 # a well-formed v2 submission: v1 receipt AND an ATOMS block that passes + covers its claim
    good = {
        "cell": "example_cell_v2gate",
        "verdict": "negative",
        "rerun": {
            "command": f"python3 {cell_rel}",
            "artifact": art,
            "checks": [{"key": "panel.case_a.slope", "expected": 0.03946, "tol": 1e-4}],
        },
        "mechanism": "the statistic is encoder-confounded; the captures lack the white-noise fingerprint.",
        "control": "the positive control reads native through the same encoder -> the method can see the signal.",
        "ATOMS": {
            "report": "example_cell_negative",
            "claims": [{"id": "naive_fp", "text": "the naive statistic gives a false positive on case A",
                        "load_bearing": True}],
            "atoms": [
                {"id": "case_a_slope", "claim": "naive_fp", "type": "value-in-artifact", "artifact": art,
                 "key": "panel.case_a.slope", "expected": 0.03946, "tol": 1e-4},
                {"id": "naive_flag", "claim": "naive_fp", "type": "value-in-artifact", "artifact": art,
                 "key": "FINDING.naive_method_false_positive", "expected": True},
            ],
        },
    }
    # v1 ok but ATOMS number poisoned -> v2 BLOCK
    v1ok_atomsbad = json.loads(json.dumps(good))
    v1ok_atomsbad["ATOMS"]["atoms"][0]["expected"] = 0.5   # real is 0.03946
    # ATOMS ok but v1 receipt missing artifact -> v2 BLOCK
    atomsok_v1bad = json.loads(json.dumps(good))
    atomsok_v1bad["rerun"].pop("artifact")
    # HYPOTHESIS-tagged -> ALLOW as hypothesis (no ATOMS required)
    hyp = {"cell": "idea", "verdict": "positive", "hypothesis": True, "note": "HYPOTHESIS awaiting-QC"}

    cases = {
        "i_v1ok_atomsok": (good, "ALLOW"),
        "ii_v1ok_atoms_poisoned": (v1ok_atomsbad, "BLOCK"),
        "iii_atomsok_v1_missing_artifact": (atomsok_v1bad, "BLOCK"),
        "iv_hypothesis": (hyp, "ALLOW"),
    }
    results, all_pass = {}, True
    for name, (sub, expect) in cases.items():
        r = fold_gate_v2(sub, base_dir=tmp)
        ok = (r["decision"] == expect)
        detail = True
        if name == "ii_v1ok_atoms_poisoned":
            detail = r["v1_decision"] == "ALLOW" and r["atoms_decision"] == "BLOCK"
        if name == "iii_atomsok_v1_missing_artifact":
            detail = r["v1_decision"] == "BLOCK" and r["atoms_decision"] == "ALLOW"
        if name == "iv_hypothesis":
            detail = r.get("as") == "hypothesis"
        if name == "i_v1ok_atomsok":
            detail = r["v1_decision"] == "ALLOW" and r["atoms_decision"] == "ALLOW"
        passed = ok and detail
        all_pass = all_pass and passed
        results[name] = {"expected": expect, "got": r["decision"], "PASS": passed,
                         "v1_decision": r.get("v1_decision"), "atoms_decision": r.get("atoms_decision"),
                         "missing": r["missing"]}
    return {"gate": "fold_gate_v2 (v1 receipt + ATOMS coverage/triviality)",
            "all_pass": all_pass, "cases": results}


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        out = _selftest()
        outdir = os.environ.get("FOLD_GATE_OUT", os.path.join(os.getcwd(), "reports", "probes"))
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "fold_gate_v2_selftest.json")
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        for name, r in out["cases"].items():
            print(f"[{'PASS' if r['PASS'] else 'FAIL'}] {name}: expected {r['expected']} got {r['got']} "
                  f"(v1={r['v1_decision']} atoms={r['atoms_decision']})")
        print(f"ALL_PASS={out['all_pass']} -> {outpath}")
        sys.exit(0 if out["all_pass"] else 1)
    execute = False
    if argv and argv[0] in ("--execute", "-x"):
        execute, argv = True, argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(2)
    src = sys.stdin.read() if argv[0] == "-" else argv[0]
    res = fold_gate_v2(src, execute=execute)
    print(json.dumps({k: v for k, v in res.items() if k not in ("v1", "atoms")}, indent=2))
    sys.exit(0 if res["decision"] == "ALLOW" else 1)
