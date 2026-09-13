"""overdet_provenance_check — automate the SHARED-IMPORT MECHANISM-IDENTITY test for cross-worktree 'over-determination'
claims. A claimed 'over-det / certified independently' is often MECHANISM-IDENTITY: the two cells import + call the SAME load-bearing
function (same math on two substrates = n_eff=1, NOT independent evidence). This module classifies a claimed over-det
from the two cells' SOURCE alone -- no execution, read the .py source text.

  overdet_provenance_check(src_a, src_b, decorr_claim=None) -> dict
    src_a, src_b: the two cells' source TEXT (or a path -- if it exists on disk it is read).
    decorr_claim: optional the claimed decorrelation number (e.g. 0.63) to check assert-vs-measure.
  Returns {shared_module_calls, cross_lane_import, shared_load_bearing, decorr_computed, verdict}.

VERDICTS:
  MECHANISM_IDENTITY: the two cells call the SAME graph_engine function (or one imports the other lane's src and calls
                       its function) -> one mechanism on two substrates, NOT independent over-det.
  DECORR_ASSERTED_NOT_MEASURED: the decorr_claim appears in the text but no correlation is COMPUTED (no corrcoef/
                       spearman/pearson/dcor of the two legs) -> recycled/stock number, not a measured decorrelation.
  LIKELY_GENUINE: no shared load-bearing call AND (if a decorr is claimed) it is computed -> plausible different-mechanism
                       over-det (still verify the DIFFERENT-BLIND-SPOT claim numerically; this checks provenance, not the
                       full over-det).
Pure python (no numpy). Composes the over-determination doctrine.
"""
from __future__ import annotations
import os
import re


def _read(s):
    if isinstance(s, str) and "\n" not in s and len(s) < 400 and os.path.exists(s):
        return open(s).read()
    return s if isinstance(s, str) else ""


def _cad_calls(text):
    """graph_engine.X functions imported/called in the text."""
    calls = set()
    for m in re.finditer(r"from\s+graph_engine\.(\w+)\s+import\s+([\w,\s]+)", text):
        for nm in m.group(2).split(","):
            nm = nm.strip()
            if nm:
                calls.add(nm)
    for m in re.finditer(r"graph_engine\.(\w+)", text):
        calls.add(m.group(1))
    # bare function calls that are clearly graph_engine deployables imported above
    return calls


# Marker for "this file imports another worktree's source tree". Override with
# $EXTERNAL_SRC_PATTERN when the worktrees are named differently.
_EXTERNAL_SRC_PATTERN = os.environ.get("EXTERNAL_SRC_PATTERN", r"[\w.-]+-[A-Z]/src")


def _cross_lane_src_import(text):
    return bool(re.search(r"sys\.path\.insert\([^)]*" + _EXTERNAL_SRC_PATTERN, text)) or \
           bool(re.search(_EXTERNAL_SRC_PATTERN, text))


def _decorr_computed(text):
    """is a correlation/decorrelation actually COMPUTED (not just stated in prose)?"""
    return bool(re.search(r"\b(corrcoef|spearmanr?|pearsonr?|np\.corrcoef|distance_correl|dcor|_spear|kendall)\b", text))


def overdet_provenance_check(src_a, src_b, decorr_claim=None):
    a = _read(src_a); b = _read(src_b)
    if not a or not b:
        return dict(verdict="INCONCLUSIVE: could not read one/both sources", shared_load_bearing=None)
    ca, cb = _cad_calls(a), _cad_calls(b)
    shared = sorted(ca & cb)
    cross = _cross_lane_src_import(a) or _cross_lane_src_import(b)
    shared_load_bearing = bool(shared) or cross
    dc = _decorr_computed(a) or _decorr_computed(b)

    decorr_flag = None
    if decorr_claim is not None:
        claim_in_text = str(decorr_claim) in a or str(decorr_claim) in b
        decorr_flag = bool(claim_in_text and not dc)     # asserted in text but NOT computed

    if shared_load_bearing:
        verdict = ("MECHANISM_IDENTITY: the two cells share a load-bearing computation (%s) -> ONE mechanism on two "
                   "substrates = n_eff=1, NOT independent over-determination. Reframe as cross-substrate REPLICATION."
                   % ("shared calls " + str(shared) if shared else "cross-worktree src import"))
    elif decorr_flag:
        verdict = ("DECORR_ASSERTED_NOT_MEASURED: the claimed decorr=%s appears in the text but NO correlation is computed "
                   "(no corrcoef/spearman/dcor of the two legs) -> recycled/stock number, not a measured decorrelation." % decorr_claim)
    else:
        verdict = ("LIKELY_GENUINE (provenance): no shared load-bearing call%s -> plausible different-mechanism over-det. "
                   "STILL verify the DIFFERENT-BLIND-SPOT claim numerically (this checks provenance, not the full over-det)."
                   % ("; decorr is computed" if (decorr_claim is not None and dc) else ""))
    return dict(shared_module_calls=shared, cross_lane_src_import=cross,
                shared_load_bearing=shared_load_bearing, decorr_computed=dc,
                decorr_asserted_not_measured=decorr_flag, verdict=verdict)


def _selftest():
    ok = tot = 0
    # (1) MECHANISM-IDENTITY: both import the same graph_engine tool (the F<->G apriori_ceiling case)
    a = "from graph_engine.apriori_identifiability_ceiling import apriori_identifiability_ceiling\nx=apriori_identifiability_ceiling(J)"
    b = "sys.path.insert(0,'/tmp/worktree-G/src')\nfrom graph_engine.apriori_identifiability_ceiling import apriori_identifiability_ceiling"
    r1 = overdet_provenance_check(a, b)
    tot += 1; ok += ("MECHANISM_IDENTITY" in r1["verdict"] and "apriori_identifiability_ceiling" in r1["shared_module_calls"])
    # (2) DECORR asserted not measured: independent code, but the claimed 0.63 is only in prose
    a2 = "import numpy as np\n# flutter SHARP vs keyhole SCATTER, decorr 0.63 = a measured over-det\nx=np.svd(A)"
    b2 = "import numpy as np\nkeyhole = poly_fit(P)"
    r2 = overdet_provenance_check(a2, b2, decorr_claim=0.63)
    tot += 1; ok += ("DECORR_ASSERTED_NOT_MEASURED" in r2["verdict"])
    # (3) LIKELY_GENUINE: independent code, decorr actually computed
    a3 = "import numpy as np\nrho=np.corrcoef(legG, legF)[0,1]  # measured decorrelation\n"
    b3 = "import numpy as np\nx=poly(P)"
    r3 = overdet_provenance_check(a3, b3, decorr_claim=0.4)
    tot += 1; ok += ("LIKELY_GENUINE" in r3["verdict"])
    # (4) no shared call, no decorr claim -> LIKELY_GENUINE
    r4 = overdet_provenance_check("import numpy as np\nx=1", "import numpy as np\ny=2")
    tot += 1; ok += ("LIKELY_GENUINE" in r4["verdict"])
    print("overdet_provenance_check selftest:")
    print("  (1) shared-import -> %s" % r1["verdict"][:55])
    print("  (2) decorr-in-prose-only -> %s" % r2["verdict"][:55])
    print("  (3) decorr-computed -> %s" % r3["verdict"][:55])
    print("  %d/%d PASS" % (ok, tot))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
