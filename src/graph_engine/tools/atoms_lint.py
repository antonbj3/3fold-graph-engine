#!/usr/bin/env python3
"""atoms_lint.py -- static lint for ATOMS blocks, run BEFORE atoms_runner.

Three defect classes made reports un-runnable by atoms_runner and had to be repaired by hand;
this lint mechanises that repair so they surface as line errors instead:
  1. every atom carried a "status" field instead of a "type" field -- atoms_runner.run_atom
     classes those as `unknown atom type None`.
  2. the ATOMS block was a BARE LIST (`[...]`, no `{"atoms": [...]}` wrapper and no top-level
     "ATOMS" key) and the entries had no "type" -- parse_atoms_block never finds the block.
  3. atoms referenced keys with BRACKET LIST INDICES ("rows[0].value") -- atoms_runner._dig
     supports DOT-numeric list indices ("stations.11.ratio") but NOT bracket syntax; a bracket key
     silently resolves to "key not present" at run time (a DEAD key that looks live in the JSON).

WHAT IT VALIDATES (STRUCTURAL only -- it never runs the judges or the commands):
  1. the block is found at all and is wrapped correctly: a dict with an "atoms" key (directly, or
     under a top-level "ATOMS" key when the report is JSON). A BARE LIST is caught.
  2. type is in atoms_runner.ATOM_TYPES -- a missing/unknown type field is caught.
  3. no bracket list-index keys in key/lhs.key/rhs.key/left.key/right.key. Dot-numeric indices are
     allowed (atoms_runner._dig supports them).
  4. referenced ARTIFACTS exist on disk and (when the artifact is readable JSON) the referenced KEY
     actually resolves in that file -- a "dead key" is caught HERE, before atoms_runner runs.
  5. a block with 0 atoms is vacuous: ALLOW on 0/0 proves nothing, so it is an error.

RUN:
  python3 atoms_lint.py <report.json|report.md>    # exit 0 = lint GREEN
  python3 atoms_lint.py --selftest                 # two-sided: the three defect classes are
 # caught, a healthy report passes
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from fold_gate import _resolve, _dig, BASE_DIR  # noqa: E402
from atoms_runner import ATOM_TYPES, parse_atoms_block  # noqa: E402

ROOT = BASE_DIR
_BRACKET_INDEX = re.compile(r"\[\s*\d+\s*\]")


def _key_fields_of(atom):
    """Every dotted-key-like field an atom might carry, paired with a human label."""
    out = []
    if "key" in atom:
        out.append(("key", atom["key"]))
    for side in ("lhs", "rhs", "left", "right"):
        v = atom.get(side)
        if isinstance(v, dict) and "key" in v:
            out.append((f"{side}.key", v["key"]))
    return out


def _artifact_fields_of(atom):
    """Every artifact/path/file reference an atom might carry, paired with the dict that also carries
    its 'key' (if any) so the key can be resolved against the artifact."""
    out = []
    if "artifact" in atom or "path" in atom:
        out.append((atom, atom.get("artifact") or atom.get("path"), atom.get("key")))
    for side in ("lhs", "rhs", "left", "right"):
        v = atom.get(side)
        if isinstance(v, dict):
            ref = v.get("artifact") or v.get("path") or v.get("file")
            if ref:
                out.append((v, ref, v.get("key")))
    return out


def lint_block(block, base_dir=ROOT, structural_errors=None):
    """block: the parsed atoms-block dict ({'atoms': [...]}), or None if unparseable.
    structural_errors: pre-existing errors found while LOCATING the block (bare-list case etc.)."""
    errors = list(structural_errors or [])
    atoms = (block or {}).get("atoms") if isinstance(block, dict) else None
    if atoms is None:
        errors.append({"severity": "ERROR", "code": "NO_ATOMS_LIST",
                        "msg": "ingen 'atoms'-lista hittad i blocket (efter inpackning)"})
        atoms = []
    if not isinstance(atoms, list):
        errors.append({"severity": "ERROR", "code": "ATOMS_NOT_LIST",
                        "msg": f"'atoms' is {type(atoms).__name__}, expected list"})
        atoms = []

    for i, atom in enumerate(atoms):
        aid = atom.get("id", f"<atom-index-{i}, SAKNAR id>") if isinstance(atom, dict) else f"<atom-index-{i}>"
        if not isinstance(atom, dict):
            errors.append({"severity": "ERROR", "code": "ATOM_NOT_DICT", "atom": aid,
                            "msg": f"atom[{i}] is {type(atom).__name__}, expected dict"})
            continue

        if "id" not in atom:
            errors.append({"severity": "ERROR", "code": "MISSING_ID", "atom": aid,
                            "msg": f"atom[{i}] has no 'id' field"})

        # ---- 1) type must be in ATOM_TYPES ------------------------------------------------------------
        t = atom.get("type")
        if t is None:
            errors.append({"severity": "ERROR", "code": "MISSING_TYPE", "atom": aid,
                            "msg": "atom has no 'type' field (atoms_runner cannot run it -- "
                                   "status-instead-of-type defect class)"})
        elif t not in ATOM_TYPES:
            errors.append({"severity": "ERROR", "code": "UNKNOWN_TYPE", "atom": aid,
                            "msg": f"type={t!r} is not one of {ATOM_TYPES}"})

        # ---- 1b) command-exit-0 requires a non-empty 'command' (fail-open class, measured: a
        # report wrote the field as 'cmd' -> the runner ran subprocess.run("") -> exit 0 -> the atom
        # PASSED without running anything, and the lint was silent) ---------------------------
        if t == "command-exit-0":
            cmd_val = atom.get("command")
            if not (isinstance(cmd_val, str) and cmd_val.strip()):
                # alias-listan HARMONISERAD med atoms_runner (atoms_omcert_v1)
                extra = [k for k in atom if k in ("cmd", "kommando", "shell", "run")]
                errors.append({"severity": "ERROR", "code": "EMPTY_COMMAND", "atom": aid,
                                "msg": "command-exit-0 atom has no non-empty 'command' field"
                                       + (f" (found {extra!r} -- the field is named 'command')" if extra else "")
                                       + " -- an empty command yields exit 0 = fail-open"})

        # ---- 1c) required fields per type (schema fail-open, measured: an atom with
        # 'json_path' instead of 'key' passed the lint and CRASHED the runner with an
        # AttributeError in the dotted-path walk -- fail-loud there, but the lint was silent) ----
        if t == "value-in-artifact" and not (atom.get("key") and atom.get("artifact")):
            errors.append({"severity": "ERROR", "code": "MISSING_FIELDS", "atom": aid,
                            "msg": "value-in-artifact requires 'artifact' + 'key' (found: "
                                   f"{sorted(k for k in atom if k not in ('type','id','why','expected','tol'))!r})"})
        if t == "inequality" and not (atom.get("lhs") is not None and atom.get("op") and atom.get("rhs") is not None):
            errors.append({"severity": "ERROR", "code": "MISSING_FIELDS", "atom": aid,
                            "msg": "inequality requires 'lhs' + 'op' + 'rhs'"})

        # ---- 2) inga listindex-nycklar (bracket-syntax) ------------------------------------
        for field_name, key in _key_fields_of(atom):
            if isinstance(key, str) and _BRACKET_INDEX.search(key):
                errors.append({"severity": "ERROR", "code": "LISTINDEX_KEY", "atom": aid,
                                "field": field_name, "value": key,
                                "msg": f"{field_name}={key!r} uses a bracket list index "
                                       "([N]) -- atoms_runner._dig() does NOT support it (it silently "
                                       "resolves to 'key not present'). Use a dot-numeric index "
                                       "instead, e.g. 'rows.0.value' "
                                       "(bracket-index defect class)."})

        # ---- 3) referenced artifacts exist + dead keys ---------------------------------
        for owner_dict, ref, key in _artifact_fields_of(atom):
            p = _resolve(ref, base_dir)
            if not p or not os.path.exists(p):
                errors.append({"severity": "ERROR", "code": "ARTIFACT_MISSING", "atom": aid,
                                "value": ref, "msg": f"referenced artifact missing on disk: {ref}"})
                continue
            if key and (_BRACKET_INDEX.search(key) is None) and p.endswith(".json"):
                try:
                    doc = json.load(open(p))
                except Exception as e:
                    errors.append({"severity": "WARN", "code": "ARTIFACT_UNREADABLE", "atom": aid,
                                    "value": ref, "msg": f"could not read {ref} as JSON: {e}"})
                    continue
                found, _val = _dig(doc, key)
                if not found:
                    errors.append({"severity": "ERROR", "code": "DEAD_KEY", "atom": aid,
                                    "value": f"{ref}#{key}",
                                    "msg": f"key {key!r} does NOT resolve in {ref} (dead key)"})

    # ---- 0 atoms = vacuous (measured: an edit emptied a list and both lint and runner returned
    # GREEN/ALLOW on 0/0 -- an empty block proves nothing) ---------------------------------
    if len(atoms) == 0:
        errors.append({"severity": "ERROR", "code": "EMPTY_ATOMS", "atom": "",
                        "msg": "0 atoms in the block -- ALLOW on 0/0 is vacuous, never green"})

    n_err = sum(1 for e in errors if e["severity"] == "ERROR")
    n_warn = sum(1 for e in errors if e["severity"] == "WARN")
    return {"n_atoms": len(atoms), "n_errors": n_err, "n_warnings": n_warn,
            "verdict": "RED" if n_err else "GREEN", "errors": errors}


def lint_report(report, base_dir=ROOT):
    """Locate + lint the ATOMS block in a report (path, JSON text, or dict). Detects the
    'bare list, no wrapper' structural failure (bare-list defect class) explicitly, since
    atoms_runner.parse_atoms_block just returns None (no-atoms-block-found) for it -- we want a
    NAMED diagnosis, not a generic 'nothing found'."""
    structural = []
    raw = report
    if isinstance(report, str) and len(report) < 4096 and os.path.exists(report):
        raw = json.load(open(report))
    elif isinstance(report, str):
        try:
            raw = json.loads(report)
        except Exception:
            raw = report  # markdown text; parse_atoms_block handles fences/sentinels

    block, src = parse_atoms_block(raw)

    if block is None and isinstance(raw, dict):
        atoms_val = raw.get("ATOMS") or raw.get("atoms")
        if isinstance(atoms_val, list):
            structural.append({"severity": "ERROR", "code": "BARE_LIST_NOT_WRAPPED",
                                "msg": "ATOMS/atoms is a BARE LIST -- atoms_runner requires "
                                       "{'atoms': [...]} (bare-list defect class); "
                                       "wrap the list under an 'atoms' key."})
            block = {"atoms": atoms_val}

    result = lint_block(block, base_dir=base_dir, structural_errors=structural)
    result["source"] = src
    return result


# ---------------------------------------------------------------------- SELFTEST (two-sided)
def selftest():
    """Two-sided proof on synthetic reports: each of the three defect classes is caught (RED) and a
    healthy report passes (GREEN)."""
    results = []
    bad = [
        ("status_instead_of_type",
         {"atoms": [{"id": "a1", "status": "PASS"}]}),
        ("bare_list_block",
         [{"id": "a1", "status": "PASS"}]),
        ("bracket_list_index",
         {"atoms": [{"id": "a1", "type": "value-in-artifact",
                     "artifact": "missing_artifact.json", "key": "rows[0].value"}]}),
        ("empty_block", {"atoms": []}),
    ]
    for name, doc in bad:
        r = lint_report(doc)
        results.append({"case": name, "expect": "RED", "got": r["verdict"],
                         "pass": r["verdict"] == "RED", "n_errors": r["n_errors"]})

    good = {"atoms": [{"id": "a1", "type": "inequality",
                       "lhs": {"value": 1.0}, "op": ">=", "rhs": {"value": 0.5}}]}
    r = lint_report(good)
    results.append({"case": "healthy_report", "expect": "GREEN", "got": r["verdict"],
                    "pass": r["verdict"] == "GREEN", "n_errors": r["n_errors"], "errors": r["errors"]})

    all_pass = all(x["pass"] for x in results)
    return {"schema": "atoms_lint_selftest", "ALL_PASS": all_pass, "cases": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        st = selftest()
        print(json.dumps(st, indent=1, ensure_ascii=False))
        return 0 if st["ALL_PASS"] else 1

    if not a.report:
        print(__doc__)
        return 2

    r = lint_report(a.report)
    print(f"atoms_lint: verdict={r['verdict']}  {r['n_atoms']} atoms, {r['n_errors']} error(s), "
          f"{r['n_warnings']} warning(s)  (source={r.get('source')})")
    for e in r["errors"]:
        print(f"  [{e['severity']}][{e['code']}] {e.get('atom', '')}: {e['msg']}")
    return 0 if r["verdict"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
