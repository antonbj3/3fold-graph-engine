#!/usr/bin/env python3
"""atoms_runner.py — run the ATOMS block of a cell-report so that QC = running the list.

WHY THIS EXISTS (the leverage, agent worktree):
  fold_gate v1 mechanized "don't take the agent's WORD — open the cited artifact and check the number".
  But it gates a *fold-submission* the harvester writes; the decomposition of a result into checkable
  atoms is still done by the QC reader, at read time, by judgement. The ATOMS standard moves that
  decomposition UPSTREAM and makes it the SUPPLIER'S duty: every cell-report carries a machine-readable
  ATOMS block {claim, check-command, expected value +/- tolerance}. QC then stops being judgement and
  becomes a lookup: run atoms_runner, read PASS/FAIL. See docs/ATOMS_STANDARD.md for the spec.

ATOM TYPES (docs/ATOMS_STANDARD.md is authoritative):
  artifact-exists: a cited file exists on disk                       (weak: existence != correctness)
  value-in-artifact: artifact[dotted.key] == expected (+/- tol)        (strength set by the tolerance band)
  command-exit-0: a runnable command returns exit 0                 (needs an assertion to be non-weak)
  inequality: artifact[key] OP {literal | artifact[key]}        (one-sided ineqs are generically weak)
  round-trip-identical: left == right, exact (values) or sha256 (files)   (tautology if left IS right)

TWO THINGS EVERY RUN REPORTS:
  1. PASS/FAIL per atom (the number is actually looked up / the command actually run).
  2. A COVERAGE + TRIVIALITY judgement, computed mechanically:
       - coverage: every load-bearing claim needs >=1 PASSING, NON-WEAK atom. A claim with no atom is,
         by the standard's definition, a HYPOTHESIS (not a result). A claim backed only by weak/trivial
         atoms is THIN (effectively a hypothesis).
       - triviality (anti-gaming): would this atom pass on a *random* result? Operationalized as a
         Monte-Carlo null-pass-rate over a documented null (uniform +/-10x the value's own scale, seed
         fixed) for value/inequality atoms; a same-source tautology check for round-trip; an
         assertion-token heuristic for command-exit-0. High null-pass-rate => the atom is vacuous.

Callable:  run_atoms(report_text_or_json, base_dir=None, execute=False, timeout=30) -> dict
CLI:       python3 scripts/atoms_runner.py <report.md|report.json|->        # Level-1 (no command exec)
           python3 scripts/atoms_runner.py --execute <report>              # also run command-exit-0 atoms
           python3 scripts/atoms_runner.py --validate                      # 4 real cases -> validation JSON
"""
import json, os, re, sys, random, hashlib, subprocess

# Build ON fold_gate v1: reuse its verified path/dig/match/token primitives rather than re-deriving them.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from fold_gate import _resolve, _dig, _match, _script_token, BASE_DIR  # noqa: E402

ATOM_TYPES = ("artifact-exists", "value-in-artifact", "command-exit-0", "inequality", "round-trip-identical")

# strength bands over the null-pass-rate (probability a RANDOM result passes the atom)
STRONG, MEDIUM, WEAK, TRIVIAL = "STRONG", "MEDIUM", "WEAK", "TRIVIAL"
_T_TRIVIAL = 0.9   # pr >= 0.9  -> vacuous, passes on ~anything
_T_WEAK = 0.5      # 0.5 <= pr < 0.9 -> low discriminating power (e.g. a coin-flip inequality / a boolean)
_T_MEDIUM = 0.1    # 0.1 <= pr < 0.5 -> medium; pr < 0.1 -> STRONG
_NULL_N = 4000
_NULL_SEED = 20260714
_NULL_K = 10.0     # null spans uniform(-K*scale, +K*scale) around the value's own magnitude

# atoms_trivial_svep_v1: the original list missed common shell-native
# assertion idioms (`test -f/-e/-d/-L/-s`, `curl -sf` fail-on-non-2xx, `grep -q` exit-code-as-check,
# `node --check` syntax-gate) -- measured across 8 reports today, atoms written as
# `command-exit-0` with these tokens were scored WEAK ("bare run, proves runnability not
# correctness") even though `test -f <missing>` / `curl -sf <404>` / `grep -q <absent-string>`
# DO exit nonzero on a wrong result -- they are genuine assertions, just not recognized by the
# heuristic's token list. Adding them is a precision fix (recognizes an assertion that was
# already there), not a threshold loosening: it can only move WEAK->MEDIUM for atoms whose
# command text already contains one of these tokens, never grant MEDIUM to a bare `python3 foo.py`.
_ASSERT_TOKENS = ("assert", "sys.exit", "--check", "!=", "==", "-eq", "-ne", "-lt", "-gt",
                  " diff ", " cmp ", "raise ", "np.testing", "allclose",
                  "test -f", "test -e", "test -d", "test -L", "test -s", "test !",
                  "-sf ", "-sf\n", "grep -q", "grep -c", "node --check")


# ---------------------------------------------------------------- block extraction
_FENCE = re.compile(r"```+\s*atoms\s*\n(.*?)```", re.S | re.I)
_SENTINEL = re.compile(r"ATOMS-BEGIN\s*(.*?)\s*ATOMS-END", re.S | re.I)


def parse_atoms_block(report):
    """Pull the ATOMS block (dict with 'atoms', optional 'claims') out of a report.

    Accepts, in order of preference:
      * a dict that already has an 'ATOMS' key (JSON report carrying the block)      -> report['ATOMS']
      * a dict that IS an atoms block (has 'atoms')                                    -> report
      * a path to a .json/.md file                                                     -> read then re-parse
      * report text with a ```atoms ...``` fenced block                                -> json.loads(fence)
      * report text with ATOMS-BEGIN... ATOMS-END sentinels                           -> json.loads(sentinel)
      * whole text is JSON with an 'ATOMS' key                                         -> that key
    Returns (block_dict_or_None, source_str).
    """
    if isinstance(report, dict):
        if "ATOMS" in report and isinstance(report["ATOMS"], dict):
            return report["ATOMS"], "dict.ATOMS"
        if "atoms" in report:
            return report, "dict"
        return None, "dict(no-atoms)"
    if not isinstance(report, str):
        return None, "unsupported"
    s = report.strip()
    if len(s) < 4096 and os.path.exists(s):
        with open(s) as f:
            return parse_atoms_block(f.read())
    m = _FENCE.search(report)
    if m:
        try:
            return json.loads(m.group(1)), "fenced```atoms"
        except Exception as e:
            return None, f"fenced-block-not-json:{type(e).__name__}"
    m = _SENTINEL.search(report)
    if m:
        try:
            return json.loads(m.group(1)), "ATOMS-BEGIN/END"
        except Exception as e:
            return None, f"sentinel-block-not-json:{type(e).__name__}"
    if s[:1] in "{[":
        try:
            obj = json.loads(s)
            return parse_atoms_block(obj)
        except Exception:
            pass
    return None, "no-atoms-block-found"


# ---------------------------------------------------------------- artifact/value helpers
def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _operand_value(op, base_dir):
    """Resolve an inequality/round-trip operand: a literal, {artifact,key}, or {file}."""
    if isinstance(op, (int, float, bool, str)) or op is None:
        return True, op, repr(op)
    if isinstance(op, dict):
        if "file" in op:
            p = _resolve(op["file"], base_dir)
            if not os.path.exists(p):
                return False, None, f"file-missing:{op['file']}"
            return True, _sha256(p), f"sha256({op['file']})"
        art = op.get("artifact")
        key = op.get("key")
        if art and key:
            p = _resolve(art, base_dir)
            if not os.path.exists(p):
                return False, None, f"artifact-missing:{art}"
            found, val = _dig(_load_json(p), key)
            if not found:
                return False, None, f"key-missing:{key}"
            return True, val, f"{art}:{key}"
    return False, None, f"unresolvable-operand:{op!r}"


_OPS = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


# ---------------------------------------------------------------- the triviality check (anti-gaming)
def _null_draws(ref):
    rng = random.Random(_NULL_SEED)
    scale = abs(ref) if (isinstance(ref, (int, float)) and not isinstance(ref, bool) and ref != 0) else 1.0
    lo, hi = -_NULL_K * scale, _NULL_K * scale
    return [rng.uniform(lo, hi) for _ in range(_NULL_N)]


def _band(pr):
    if pr is None:
        return WEAK
    if pr >= _T_TRIVIAL:
        return TRIVIAL
    if pr >= _T_WEAK:
        return WEAK
    if pr >= _T_MEDIUM:
        return MEDIUM
    return STRONG


_TOL_ALIAS = ("tol", "tolerance", "tolerans", "atol")
_KNOWN_ATOM_KEYS = {"type", "id", "claim", "artifact", "key", "expected", "value", "op", "lhs",
                    "rhs", "left", "right", "command", "cmd", "file", "load_bearing", "note",
                    "why", "desc", "description"} | set(_TOL_ALIAS)


def _tol_of(atom):
    """The tolerance, whichever of the common names the author used.

    SILENT CONTRACT TRAP (measured twice, by two INDEPENDENT authors): the schema read ONLY `tol`,
    so an atom written with `tolerance` got eff_tol ~1e-6 -- a two-sided band fell back silently to
    exact equality and the atom FAILED even though the value was well inside the band. The same
    class hit `cmd` vs `command` (subprocess.run("") returns 0 -> the atom passed without running
    anything, a fail-open). An unknown key must never silently change an atom's semantics."""
    for k in _TOL_ALIAS:
        v = atom.get(k)
        if v is not None:
            return v
    return None


def unknown_atom_keys(atom):
    """Keys the gate does NOT understand -- reported so a misspelled key is visible instead of
    silently changing the atom's strength."""
    return sorted(k for k in atom.keys() if k not in _KNOWN_ATOM_KEYS)


def _atom_sources(atom):
    """Every artifact path an atom reads (whatever the atom type)."""
    out = []
    for key in ("artifact",):
        if isinstance(atom.get(key), str):
            out.append(atom[key])
    for key in ("lhs", "rhs", "left", "right"):
        v = atom.get(key)
        if isinstance(v, dict) and isinstance(v.get("artifact"), str):
            out.append(v["artifact"])
    return out


def _source_is_self(src, base_dir, self_sha, self_obj):
    """Content identity: is the artifact `src` CONTENT-IDENTICAL to the report being certified?
    Catches self-reference EVEN in contents mode (where the path is absent) -- the finding was that
    the runner's path guard was INERT when the runner was called with the report CONTENTS."""
    try:
        raw = open(_resolve(src, base_dir), "rb").read()
    except Exception:
        return False
    if self_sha is not None and hashlib.sha256(raw).hexdigest() == self_sha:
        return True
    if self_obj is not None:
        try:
            if json.loads(raw.decode("utf-8", "replace")) == self_obj:
                return True
        except Exception:
            pass
    return False


def atom_strength(atom, base_dir, self_path=None, self_sha=None, self_obj=None):
    """Would a RANDOM result pass this atom? Returns {strength, null_pass_rate, reason}.

    Independent of whether the *actual* value passes: it measures the atom's discriminating power.

    The tautology guard used to exist ONLY for round-trip-identical. An atom whose artifact is THE
    REPORT ITSELF was therefore read as STRONG (null_pass_rate=0.0) and made its claim COVERED --
    i.e. the report proved itself. MEASURED blast radius: 2023 of 2391 ATOMS-bearing reports (85 %)
    carry at least one such atom. The rule is the same as for round-trip: a quantity cannot be its
    own witness.
    """
    srcs = _atom_sources(atom)
    if srcs:
        by_path = bool(self_path) and all(
            os.path.abspath(_resolve(s, base_dir)) == os.path.abspath(self_path) for s in srcs)
        # contents-mode fix: the path guard was INERT
        # when the runner was called with the report CONTENTS. Content identity catches it: an atom
        # whose source is CONTENT-IDENTICAL to the report itself is self-reference in either mode.
        by_content = (self_sha is not None or self_obj is not None) and all(
            _source_is_self(s, base_dir, self_sha, self_obj) for s in srcs)
        if by_path or by_content:
            return {"strength": TRIVIAL, "null_pass_rate": 1.0,
                    "reason": ("SELF-REFERENCE: the atom's only source is the report itself -- it tests its "
                               "own statement against itself and can never fail"
                               + ("; via content identity in contents mode" if by_content and not by_path else "") + ")")}
    t = atom.get("type")
    if t == "artifact-exists":
        return {"strength": WEAK, "null_pass_rate": None,
                "reason": "existence proves the artifact was produced, not that its numbers are correct "
                          "(necessary, not sufficient)"}
    if t == "command-exit-0":
        cmd = str(atom.get("command", ""))
        has_assert = any(tok in cmd for tok in _ASSERT_TOKENS)
        return {"strength": MEDIUM if has_assert else WEAK, "null_pass_rate": None,
                "reason": ("command carries an assertion/comparison token -> it can exit nonzero on wrong "
                           "data" if has_assert else
                           "bare run: exits 0 as long as it does not crash -> proves runnability, not "
                           "correctness (add an assert/--check to raise strength)")}
    if t == "value-in-artifact":
        expected = atom.get("expected", atom.get("value"))
        if isinstance(expected, bool):
            return {"strength": WEAK, "null_pass_rate": 0.5,
                    "reason": "boolean claim: a random result matches with p=0.5 (1 bit of evidence)"}
        if isinstance(expected, (int, float)):
            tol = _tol_of(atom)
            eff_tol = tol if tol is not None else (1e-6 + 1e-6 * abs(expected))
            draws = _null_draws(expected)
            pr = sum(1 for x in draws if abs(x - expected) <= eff_tol) / len(draws)
            return {"strength": _band(pr), "null_pass_rate": round(pr, 5),
                    "reason": f"tol band +/-{eff_tol:g} vs null spread +/-{_NULL_K}x|value|; "
                              f"{pr*100:.2f}% of random results land in band"}
        # categorical exact-match on a string/other
        return {"strength": MEDIUM, "null_pass_rate": None,
                "reason": "categorical exact-match; strength depends on label cardinality (assumed >2)"}
    if t == "inequality":
        # atoms_runner_falsy_lhs_v1: `atom.get("lhs") or {...}` treated an explicitly
        # provided falsy literal (0, [], "", False) as "not provided" and silently fell back to the
        # legacy {artifact,key} pair -- either unresolvable (None/None -> wrong-reason FAIL) or, when
        # stray top-level "artifact"/"key" fields happened to be present, a silently WRONG value. Fix:
        # explicit `is None` check so only an actually-absent lhs triggers the legacy fallback.
        lhs_ref = atom.get("lhs")
        if lhs_ref is None:
            lhs_ref = {"artifact": atom.get("artifact"), "key": atom.get("key")}
        okl, lhs, _ = _operand_value(lhs_ref, base_dir)
        okr, rhs, _ = _operand_value(atom.get("rhs"), base_dir)
        op = atom.get("op")
        if not (okl and okr and op in _OPS and isinstance(lhs, (int, float)) and isinstance(rhs, (int, float))):
            return {"strength": WEAK, "null_pass_rate": None,
                    "reason": "inequality operands not both numeric/resolvable -> cannot score, treat as weak"}
        draws = _null_draws(lhs)
        pr = sum(1 for x in draws if _OPS[op](x, rhs)) / len(draws)
        return {"strength": _band(pr), "null_pass_rate": round(pr, 5),
                "reason": f"{pr*100:.2f}% of random LHS values satisfy 'x {op} {rhs:g}' "
                          f"(one-sided bounds against a loose rhs are generically weak)"}
    if t == "round-trip-identical":
        left, right = atom.get("left"), atom.get("right")
        if isinstance(left, dict) and left == right:
            return {"strength": TRIVIAL, "null_pass_rate": 1.0,
                    "reason": "TAUTOLOGY: left and right resolve to the SAME source -> compares a thing to "
                              "itself, passes always"}
        return {"strength": STRONG, "null_pass_rate": round(1.0 / _NULL_N, 6),
                "reason": "exact identity of two independently-produced operands is a strong constraint "
                          "(random operands are ~never identical)"}
    return {"strength": WEAK, "null_pass_rate": None, "reason": f"unknown atom type {t!r}"}


# ---------------------------------------------------------------- run one atom
def run_atom(atom, base_dir, execute=False, timeout=30, self_path=None, self_sha=None, self_obj=None):
    out = {"id": atom.get("id"), "claim": atom.get("claim"), "type": atom.get("type")}
    t = atom.get("type")
    if t not in ATOM_TYPES:
        out.update(ok=False, reason=f"unknown atom type {t!r} (allowed: {ATOM_TYPES})")
        out["strength"] = WEAK
        return out

    if t == "artifact-exists":
        p = _resolve(atom.get("path") or atom.get("artifact"), base_dir)
        exists = bool(p and os.path.exists(p))
        out.update(ok=exists, actual=p, reason=None if exists else "file not found")

    elif t == "value-in-artifact":
        art = _resolve(atom.get("artifact"), base_dir)
        key = atom.get("key")
        expected = atom.get("expected", atom.get("value"))
        tol = _tol_of(atom)
        out["expected"] = expected
        if not art or not os.path.exists(art):
            out.update(ok=False, actual=None, reason=f"artifact not found: {atom.get('artifact')}")
        else:
            found, actual = _dig(_load_json(art), key)
            out["actual"] = actual
            if not found:
                out.update(ok=False, reason=f"key not present: {key}")
            elif _match(expected, actual, tol):
                out.update(ok=True)
            else:
                try:
                    out["diff"] = abs(expected - actual)
                except Exception:
                    out["diff"] = f"{expected!r} != {actual!r}"
                out.update(ok=False, reason="mismatch")

    elif t == "command-exit-0":
        cmd = str(atom.get("command", "")).strip()
        # FAIL-OPEN: an atom written with the key `cmd` instead of `command` produced an empty
        # string, and `subprocess.run("")` returns 0 -- the atom PASSED without running anything.
        # The class was documented but no gate stopped it, so it kept certifying reports. An atom
        # without a command is not satisfied, it is UNWRITTEN.
        tok = _script_token(cmd)
        if not cmd:
            # the alias list is HARMONISED with atoms_lint: the lint looked for
            # ("cmd","kommando","run") and the runner for ("cmd","kommando","shell"), so an atom
            # written with "run" got a diagnosis without the key name here, and one with "shell" got
            # it in the lint. Both lists are now identical; the diagnosis is the whole value of a
 # fail-closed atom.
            alias = next((k for k in ("cmd", "kommando", "shell", "run") if atom.get(k)), None)
            out.update(ok=False, reason=(
                "empty command: the atom has no 'command' field"
                + (f" (found {alias!r} -- rename the key to 'command')" if alias else "")))
        elif tok and not os.path.exists(_resolve(tok, base_dir)):
            out.update(ok=False, reason=f"script not found: {tok}")
        elif not execute:
            out.update(ok=None, status="not-run (pass --execute to run command-exit-0 atoms)")
        else:
            try:
                proc = subprocess.run(cmd, shell=True, cwd=base_dir, timeout=timeout,
                                      capture_output=True, text=True)
                out.update(ok=(proc.returncode == 0), exit_code=proc.returncode,
                           reason=None if proc.returncode == 0
                           else f"exit {proc.returncode}: {proc.stderr.strip()[-160:]!r}")
            except subprocess.TimeoutExpired:
                out.update(ok=False, reason=f"exceeded {timeout}s -> RESTRUCTURE to a cheaper assertion")

    elif t == "inequality":
        # same falsy-literal fix as atom_strength above -- see atoms_runner_falsy_lhs_v1.
        lhs_ref = atom.get("lhs")
        if lhs_ref is None:
            lhs_ref = {"artifact": atom.get("artifact"), "key": atom.get("key")}
        okl, lhs, lsrc = _operand_value(lhs_ref, base_dir)
        okr, rhs, rsrc = _operand_value(atom.get("rhs"), base_dir)
        op = atom.get("op")
        out.update(lhs=lhs, rhs=rhs, op=op, lhs_src=lsrc, rhs_src=rsrc)
        if not okl or not okr:
            out.update(ok=False, reason=f"operand unresolvable (lhs:{lsrc} rhs:{rsrc})")
        elif op not in _OPS:
            out.update(ok=False, reason=f"unknown op {op!r} (allowed {list(_OPS)})")
        else:
            try:
                out.update(ok=bool(_OPS[op](lhs, rhs)))
            except TypeError as e:
                out.update(ok=False, reason=f"type error comparing {lhs!r} {op} {rhs!r}: {e}")

    elif t == "round-trip-identical":
        okl, lv, lsrc = _operand_value(atom.get("left"), base_dir)
        okr, rv, rsrc = _operand_value(atom.get("right"), base_dir)
        out.update(left_src=lsrc, right_src=rsrc, left=lv, right=rv)
        if not okl or not okr:
            out.update(ok=False, reason=f"operand unresolvable (left:{lsrc} right:{rsrc})")
        else:
            out.update(ok=(lv == rv), reason=None if lv == rv else "left != right")

    st = atom_strength(atom, base_dir, self_path=self_path, self_sha=self_sha, self_obj=self_obj)
    unk = unknown_atom_keys(atom)
    if unk:
        out["unknown_keys"] = unk
    out.update(strength=st["strength"], null_pass_rate=st["null_pass_rate"], strength_reason=st["reason"])
    return out


# ---------------------------------------------------------------- run a whole block
def run_atoms(report, base_dir=None, execute=False, timeout=30):
    base_dir = base_dir or BASE_DIR
    # The self-reference guard was INERT via the CLI: parse_atoms_block reads the file and
    # RE-PARSES the contents, so `src` becomes "dict.ATOMS" -- not the path. Without it the guard
    # lost its reference point and never fired on the most common call path. The original path must
    # therefore be captured HERE, before parsing.
    _self_path = None
    _self_sha = None
    _self_obj = None
    if isinstance(report, dict):
        _self_obj = report
    elif isinstance(report, str):
        _cand = report.strip()
        if len(_cand) < 4096 and os.path.isfile(_cand):
            _self_path = os.path.abspath(_cand)
            try:
                _raw = open(_self_path, "rb").read()
                _self_sha = hashlib.sha256(_raw).hexdigest()
                _self_obj = json.loads(_raw.decode("utf-8", "replace"))
            except Exception:
                pass
        else:
            # contents mode: the report CONTENTS are the witness surface -> bind the guard to the contents
            _self_sha = hashlib.sha256(report.encode("utf-8", "replace")).hexdigest()
            try:
                _self_obj = json.loads(report)
            except Exception:
                pass
    block, src = parse_atoms_block(report)
    if block is None:
        return {"decision": "BLOCK", "verdict": "NO_ATOMS_BLOCK", "source": src,
                "missing": ["no ATOMS block found in report (need a ```atoms fenced block, an ATOMS-BEGIN/END "
                            "sentinel pair, or a dict with an 'ATOMS' key)"],
                "atoms": [], "coverage": [], "triviality_flags": []}

    # allow the block to pin its own base_dir (relative to the runner base)
    if block.get("base_dir"):
        base_dir = _resolve(block["base_dir"], base_dir)

    atoms = block.get("atoms") or []
    claims = block.get("claims") or []
    # atoms_runner_tomhetshal_v1: an ATOMS block with
    # BOTH atoms[] and claims[] empty asserts nothing yet fell through the coverage/missing checks
    # below as vacuously ALLOW/GREEN (ran=[], coverage=[], missing=[] -> no reason to BLOCK). That is
    # fail-open in the judge itself -- measured in prod on reports/probes/feem_b1_master_v6_6.json,
    # _v6_11.json, gpu_lbm_utilization_selftest.json. A claims-only-empty-atoms block is NOT this hole
    # (already correctly falls to UNBACKED_HYPOTHESIS/RED below); only the fully-empty case is fixed here.
    if not atoms and not claims:
        return {"decision": "BLOCK", "verdict": "EMPTY_ATOMS_BLOCK", "source": src,
                "missing": ["ATOMS block present but both atoms[] and claims[] are empty -- an empty "
                            "block asserts nothing and cannot be ALLOW/GREEN (atoms_runner_tomhetshal_v1)"],
                "n_atoms": 0, "n_pass": 0, "n_fail": 0, "n_notrun": 0,
                "atoms": [], "coverage": [], "triviality_flags": []}
    # The emptiness fix above closed ONLY the "both empty" case. The larger hole remained: atoms[]
    # WITHOUT claims[] -> by_claim/coverage becomes [] -> missing becomes [] -> ALLOW/GREEN. That is,
    # a report could run 30 atoms without COVERAGE-TESTING A SINGLE statement and still certify
    # green. MEASURED blast radius: 1341 of 2391 ATOMS-bearing reports (56 %) have exactly this
    # shape. Atoms without statements are measurement without a conclusion.
    if atoms and not claims:
        return {"decision": "BLOCK", "verdict": "NO_CLAIMS_BLOCK", "source": src,
                "missing": [f"the ATOMS block has {len(atoms)} atom(s) but NO claims[] -- 0 statements "
                            f"coverage-tested; atoms without statements are measurement without a "
                            f"conclusion and cannot be ALLOW/GREEN"],
                "n_atoms": len(atoms), "n_pass": 0, "n_fail": 0, "n_notrun": len(atoms),
                "atoms": [], "coverage": [], "triviality_flags": []}
    ran = [run_atom(a, base_dir, execute=execute, timeout=timeout, self_path=_self_path,
                    self_sha=_self_sha, self_obj=_self_obj)
           for a in atoms]

    non_weak = {STRONG, MEDIUM}
    # ---- triviality flags (anti-gaming) ----
    triviality_flags = [{"id": r["id"], "type": r["type"], "strength": r["strength"],
                         "null_pass_rate": r.get("null_pass_rate"), "why": r.get("strength_reason")}
                        for r in ran if r["strength"] == TRIVIAL]

    # ---- coverage per load-bearing claim ----
    by_claim = {}
    for r in ran:
        by_claim.setdefault(r.get("claim"), []).append(r)
    coverage = []
    for c in claims:
        if not c.get("load_bearing", True):
            continue
        cid = c.get("id")
        crs = by_claim.get(cid, [])
        strong = [r for r in crs if r["strength"] in non_weak]
        strong_pass = [r for r in strong if r["ok"] is True]
        strong_fail = [r for r in strong if r["ok"] is False]
        all_trivial = bool(crs) and all(r["strength"] == TRIVIAL for r in crs)
        if not crs:
            verdict = "UNBACKED_HYPOTHESIS"   # a claim with no atom is, by the standard, a hypothesis
        elif all_trivial:
            verdict = "TRIVIAL_ONLY"
        elif not strong:
            verdict = "THIN"                  # only weak atoms -> effectively a hypothesis
        elif strong_pass:
            verdict = "COVERED"
        elif strong_fail:
            verdict = "REFUTED"               # a non-weak atom for this claim FAILS
        else:
            verdict = "PENDING"               # strong atom present but not-run (command not executed)
        coverage.append({"claim": cid, "text": c.get("text"), "verdict": verdict,
                         "n_atoms": len(crs), "n_nonweak_pass": len(strong_pass),
                         "atom_ids": [r["id"] for r in crs]})

    # atoms that were RUN and FAILED (command-exit-0 not-run => ok is None, not a fail)
    failed = [r for r in ran if r["ok"] is False]
    bad_coverage = [c for c in coverage if c["verdict"] != "COVERED"]

    missing = []
    for r in failed:
        d = f" diff={r['diff']}" if r.get("diff") is not None else ""
        missing.append(f"atom[{r['id']}] {r['type']} FAIL: {r.get('reason')}"
                       + (f" (expected={r.get('expected')!r} actual={r.get('actual')!r}{d})"
                          if r["type"] == "value-in-artifact" else ""))
    for c in bad_coverage:
        missing.append(f"coverage[{c['claim']}]: {c['verdict']} "
                       f"({c['n_atoms']} atom(s), {c['n_nonweak_pass']} non-weak passing) "
                       f"-- text: {c['text']!r}")
    for tf in triviality_flags:
        missing.append(f"triviality[{tf['id']}]: {tf['strength']} "
                       f"(null_pass_rate={tf['null_pass_rate']}) -- {tf['why']}")

    decision = "ALLOW" if not missing else "BLOCK"
    return {"decision": decision,
            "verdict": "GREEN" if decision == "ALLOW" else "RED",
            "source": src, "base_dir": base_dir,
            "n_atoms": len(ran), "n_pass": sum(1 for r in ran if r["ok"] is True),
            "n_fail": len(failed), "n_notrun": sum(1 for r in ran if r["ok"] is None),
            "missing": missing, "atoms": ran, "coverage": coverage,
            "triviality_flags": triviality_flags}


# ---------------------------------------------------------------- CLI
def _print(res):
    print(f"decision: {res['decision']}  verdict={res['verdict']}  "
          f"(atoms {res.get('n_pass')}/{res.get('n_atoms')} pass, {res.get('n_notrun',0)} not-run)")
    for r in res.get("atoms", []):
        flag = "ok" if r["ok"] is True else ("--" if r["ok"] is None else "FAIL")
        print(f"  [{flag}][{r['strength']}] {r['id']} ({r['type']})"
              + (f" -> {r.get('reason')}" if r.get("reason") else ""))
    for c in res.get("coverage", []):
        print(f"  coverage {c['claim']}: {c['verdict']}")
    for m in res.get("missing", []):
        print(f"  MISSING: {m}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--validate":
        from atoms_validation import validate  # companion harness
        out = validate()
        sys.exit(0 if out["all_pass"] else 1)
    # BUGFIX (hub_hardening_2026_07_26, atoms_hygiene_v1): --execute/-x used to be recognized ONLY at
    # argv[0]. `atoms_runner.py <report.json> --execute` (flag AFTER the file path -- the natural order,
    # and the order every doc example in this repo avoids but a human/agent typing it by hand reaches for)
    # silently left execute=False: every command-exit-0 atom then reports ok=None ("not-run"), coverage for
    # any claim backed only by such atoms degrades to PENDING, and the run BLOCKs/REDs -- indistinguishable
    # from a real command failure unless you notice the CLI, not the command, never ran. Fix: recognize the
    # flag anywhere in argv (order-independent), matching how every other flag in this repo's CLIs behaves.
    execute = False
    if "--execute" in argv or "-x" in argv:
        execute = True
        argv = [a for a in argv if a not in ("--execute", "-x")]
    if not argv:
        print(__doc__)
        sys.exit(2)
    src = sys.stdin.read() if argv[0] == "-" else argv[0]
    res = run_atoms(src, execute=execute)
    _print(res)
    sys.exit(0 if res["decision"] == "ALLOW" else 1)
