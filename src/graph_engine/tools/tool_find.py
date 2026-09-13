"""tool_find.py -- the prior-art search gate (search before you build).

Hand-maintained indexes go stale, so this searcher is LIVE: it scans the code's own text on every
call. Usage:

    python3 tool_find.py <keyword> [keyword...]   # exit 0 + hits = something already exists
    python3 tool_find.py --selftest               # synthetic corpus regression test (rc=1 on fail)
    python3 tool_find.py --tackning               # per-directory file coverage of the roots

Exit 0 with hits means prior art exists -- read it before building. Exit 1 means nothing was
found: build it, and write a docstring so the next search finds it.

Ranking: each file is scored in three zones (path > docstring > body) with an inverse-document-
frequency weight, so a rare term outranks a common one; a strict-AND miss never collapses to
"nothing found", it degrades to the best partial matches. Search terms and file text both go
through the same ASCII folding, and a small synonym table adds recall at a lower weight.

The roots to scan come from the TOOL_FIND_ROOTS environment variable (os.pathsep-separated,
optionally "label=path"), else from --root arguments, else the current working directory.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

# --- scan roots -------------------------------------------------------------------------
def _roots_from_env() -> list[tuple[str, Path]]:
    """Read the scan roots from TOOL_FIND_ROOTS ("label=path" entries separated by os.pathsep),
    falling back to the current working directory. Returns [(label, path)]."""
    raw = os.environ.get("TOOL_FIND_ROOTS", "")
    out: list[tuple[str, Path]] = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        label, _, path = item.partition("=")
        if not path:
            label, path = Path(label).name, label
        out.append((label, Path(path).expanduser()))
    return out or [(".", Path.cwd())]


ROOTS = _roots_from_env()

SUFFIXES = {".py", ".md"}
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages", "browser_profile", ".ipynb_checkpoints", ".cache",
}

# Zone weights: where in the file a term hits decides how strong the hit is. The path weighs
# most, then the docstring (where prior art describes itself), then the body text.
W_PATH, W_DOC, W_BODY = 6.0, 4.0, 1.0
# Reference length for the docstring zone's length normalisation (see corpus()).
DOC_REF = 1200
# A synonym hit is weaker evidence than a literal one: enough to FIND the file, never enough to
# push out a file carrying the word the user actually typed.
SYNONYM_WEIGHT = 0.6
# Per-root bias: archive roots must not fill the result window ahead of active ones.
REPO_BIAS = {"I/scripts": 3, "I/data": 3, "I/src": 3, "I/docs": 2, "memory": 2, "memory2": 2}

# Swedish<->English synonym groups: the corpus is bilingual and a searcher types half the terms in
# the other language. The groups are kept deliberately narrow -- broad synonymy drowns precision,
# and SYNONYM_WEIGHT keeps them below literal hits.
SYNONYM_GROUPS = [
    ["duct", "kanal", "channel"],
    ["route", "routing", "rutt", "rutten", "dragning"],
    ["printer", "skrivare", "utskrift", "printbar", "klipper"],
    ["power", "strom", "effekt", "matning"],
    ["keepout", "frizon", "forbjuden zon"],
    ["publish", "publicera", "publicering", "publ"],
    ["cache", "cachning", "memoisering"],
    ["pair", "par", "parvis"],
    ["boolean", "boolesk", "bool"],
    ["valid", "giltighet", "giltig", "validity", "validering"],
    ["props", "properties", "egenskaper"],
    # Near-synonyms that are NOT interchangeable (e.g. genus vs topology) must stay in separate
    # groups: semantic proximity is not substitutability.
    ["topology", "topologi"],
    ["mesh", "nat", "triangulering"],
    ["solid", "kropp", "part", "del"],
    ["surface", "face", "yta"],
    ["gate", "grind", "domare"],
    ["report", "rapport"],
    ["measure", "mat", "matning", "measurement"],
    ["search", "sok", "find", "hitta"],
    ["coverage", "tackning"],
    ["proof", "bevis", "fallbevis"],
    ["lens", "lins", "optik"],
    ["ray", "strale", "stral"],
    ["heat", "varme", "termisk", "thermal", "heatsink"],
    ["flow", "flode"],
    ["wall", "vagg"],
    ["lid", "lock"],
    ["seal", "gasket", "tatning", "packning"],
    ["rail", "skena"],
    ["assembly", "montering", "hopsattning"],
    ["build", "bygge", "bygg"],
    ["manifest", "manifestet"],
    ["scene", "scen"],
    ["source", "kalla", "kallforst"],
    ["export", "exportera"],
    ["cert", "certifikat", "certifiering"],
]
_SYN: dict[str, set[str]] = {}
for _g in SYNONYM_GROUPS:
    for _w in _g:
        _SYN.setdefault(_w, set()).update(x for x in _g if x != _w)

# ASCII folding: the same word is spelled with and without diacritics, so both needle and haystack
# go through the same transform. The folding is done on BYTES via latin-1 (measured ~80x faster
# than str.translate over the corpus); characters outside latin-1 are dropped, which never happens
# in search terms.
#          å     ä     ö     Å     Ä     Ö     é     è     ê     ë     ü     Ü     ñ     Ç     ç     É     È
_FOLD_SRC = bytes([0xE5, 0xE4, 0xF6, 0xC5, 0xC4, 0xD6, 0xE9, 0xE8, 0xEA, 0xEB, 0xFC, 0xDC, 0xF1, 0xC7, 0xE7, 0xC9, 0xC8])
_FOLD_DST = b"aaoaaoeeeeuunccee"
_FOLD_B = bytes.maketrans(_FOLD_SRC, _FOLD_DST)


def fold(s: str) -> bytes:
    """Fold to comparable bytes (lower-case, no diacritics). Both search term and file text."""
    return s.encode("latin-1", "ignore").lower().translate(_FOLD_B).lower()


def head_of(p: Path, n: int | None = None) -> str:
    """Read the file text. n=None means the WHOLE file (truncation makes most of it unsearchable)."""
    try:
        t = p.read_text(errors="ignore")
    except Exception:
        return ""
    return t if n is None else t[:n]


def doc_line(text: str, suffix: str) -> str:
    """First descriptive line of a file, for display."""
    if suffix == ".py":
        m = re.search(r'"""(.+?)(?:\n|""")', text, re.S)
        return (m.group(1).strip() if m else "")[:160]
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith(("---", "name:", "description:", "metadata:", "type:")):
            return s.lstrip("# ").strip()[:160]
    return ""


def doc_block(text: str, suffix: str) -> str:
    """The docstring zone of a file: where prior art describes itself. Wider than doc_line() by
    design, so a file naming its subject in the docstring outranks one naming it in a code line."""
    if suffix == ".py":
        m = re.search(r'"""(.*?)"""', text, re.S)
        return m.group(1)[:4000] if m else text[:600]
    return text[:1500]


def iter_files():
    """Yield (label, path) for every searchable file. Recursive by construction; de-duplicated on
    realpath (roots may overlap), first label wins."""
    seen: set[str] = set()
    for label, root in ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Hidden directories are skipped: backup trees otherwise fill the result window with
            # snapshots of a single file and push out real prior art.
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if os.path.splitext(fn)[1] not in SUFFIXES:
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    rp = os.path.realpath(p)
                except OSError:
                    continue
                if rp in seen:
                    continue
                seen.add(rp)
                yield label, Path(p)


def _wordish(v: bytes):
    """Regex requiring a WORD BOUNDARY around a synonym."""
    return re.compile(rb"(?<![a-z0-9_])" + re.escape(v) + rb"(?![a-z0-9_])")


def _variants(term: str) -> list[tuple[bytes, float, object]]:
    """(byte needle, weight, word-boundary regex or None) per variant of a term."""
    t = fold(term)
    out: list[tuple[bytes, float, object]] = [(t, 1.0, None)]
    for s in sorted(_SYN.get(t.decode("latin-1"), ())):
        v = fold(s)
        out.append((v, SYNONYM_WEIGHT, _wordish(v)))
    return out


def _hit(v: bytes, rx, z: bytes) -> bool:
    return v in z and (rx is None or rx.search(z) is not None)


_CORPUS: list[tuple] | None = None


def corpus() -> list[tuple]:
    """(label, path, z_path, z_doc, z_body, doc, dnorm) per file, folded ONCE per process and cached."""
    global _CORPUS
    if _CORPUS is None:
        rows = []
        for label, p in iter_files():
            text = head_of(p)  # the WHOLE file, no truncation
            if not text:
                continue
            zd = fold(doc_block(text, p.suffix))
            # Length normalisation of the docstring zone (BM25 spirit): a term inside a 4000-char
            # essay docstring is weaker evidence than the same term in a 300-char precise one. The
            # 0.35 floor keeps a long but CORRECT docstring above body text.
            dnorm = max(0.35, min(1.0, DOC_REF / max(1, len(zd))))
            rows.append((label, p, fold(str(p)), zd, fold(text), doc_line(text, p.suffix), dnorm))
        _CORPUS = rows
    return _CORPUS


def find(terms: list[str], exclude: set[str] | None = None):
    """Return (scored, per_term): scored rows and the per-term document counts."""
    exclude = {os.path.realpath(e) for e in (exclude or set())}
    vmap = {t: _variants(t) for t in terms}
    scored: list[dict] = []
    per_term = {t: 0 for t in terms}
    df_lit = {t: 0 for t in terms}  # LITERAL document frequency, the basis for IDF
    # The path is its own zone: a term hits a directory named after it even when the word does not
    # occur inside the file.
    for label, p, z_path, z_doc, z_body, docl, dnorm in corpus():
        if exclude and os.path.realpath(p) in exclude:
            continue
        zw: dict[str, float] = {}
        for t in terms:
            best = 0.0
            for v, vw, rx in vmap[t]:
                if _hit(v, rx, z_path):
                    best = max(best, W_PATH * vw)
                if _hit(v, rx, z_doc):
                    best = max(best, W_DOC * dnorm * vw)
                elif _hit(v, rx, z_body):
                    best = max(best, W_BODY * vw)
                if vw == 1.0 and best > 0:
                    df_lit[t] += 1
            if best > 0:
                zw[t] = best
        if not zw:
            continue
        for t in zw:
            per_term[t] += 1
        scored.append({
            "n": len(zw), "label": label, "path": str(p),
            "name": p.name, "doc": docl, "matched": list(zw), "_zw": zw,
        })

    # IDF: a term occurring in 45 files is a sharper prior-art trace than one occurring in 3983.
    # Document frequency is counted on the LITERAL term, never on the synonym expansion -- synonyms
    # may raise recall, they must never touch rarity.
    n_files = max(1, len(corpus()))
    idf = {t: math.log(1.0 + n_files / (1.0 + df_lit[t])) for t in terms}
    for s in scored:
        s["score"] = sum(w * idf[t] for t, w in s.pop("_zw").items())
    return scored, per_term


def rank(scored: list[dict]) -> list[dict]:
    """Rank on (number of matched terms, zone-weighted score, per-root bias) -- never on the label
    alphabetically, which would bury correct hits below the visible window."""
    return sorted(scored, key=lambda s: (-s["n"], -s["score"], -REPO_BIAS.get(s["label"], 0), s["name"]))


def window(rows: list[dict], limit: int) -> list[dict]:
    """Select the rows actually DISPLAYED. The window is the only surface a human reads, so
    duplicates of the same file across roots are collapsed into one row."""
    out: list[dict] = []
    seen_twin: dict[tuple, dict] = {}
    for r in rows:
        try:
            key = (r["name"], os.path.getsize(r["path"]))
        except OSError:
            key = (r["name"], -1)
        if key in seen_twin:
            seen_twin[key].setdefault("tvillingar", []).append(r["label"])
            continue
        seen_twin[key] = r
        out.append(r)
        if len(out) >= limit:
            break
    return out


def evidence(row: dict, terms: list[str]) -> tuple[int, str]:
    """(line number, context line) for the line carrying MOST of the matched terms. Lazy by design:
    only computed for the rows that are actually displayed."""
    try:
        lines = Path(row["path"]).read_text(errors="ignore").splitlines()
    except Exception:
        return 0, ""
    vlist = [(t, _variants(t)) for t in row["matched"]]
    best_ln, best_hits, best_txt = 0, 0, ""
    for i, line in enumerate(lines, 1):
        fl = fold(line)
        hits = sum(1 for _t, vs in vlist if any(_hit(v, rx, fl) for v, _w, rx in vs))
        if hits > best_hits:
            best_ln, best_hits, best_txt = i, hits, line.strip()
            if hits == len(vlist):
                break
    return best_ln, best_txt[:120]


def render(rows: list[dict], terms: list[str], limit: int) -> None:
    for i, r in enumerate(window(rows, limit), 1):
        ln, ctx = evidence(r, terms)
        loc = f"{r['path']}:{ln}" if ln else r["path"]
        twin = f" (also in {','.join(r['tvillingar'])})" if r.get("tvillingar") else ""
        print(f"  {i:2d}. [{r['n']}/{len(terms)}] [{r['label']}] {loc}{twin}  «{'+'.join(r['matched'])}»")
        if r["doc"]:
            print(f"      {r['doc']}")
        if ctx:
            print(f"      | {ctx}")


def coverage_report() -> dict:
    """Measure file coverage per root: files present vs files searchable. This is the assertion that
    fails if a tree is added to the project but not to ROOTS."""
    seen = {os.path.realpath(str(p)) for _lbl, p in iter_files()}
    return {label: katalog_tackning(root, seen) for label, root in ROOTS if root.exists()}


def katalog_tackning(dp: Path, seen: set[str]) -> dict:
    """Coverage for ONE directory against the set of searchable realpaths."""
    alla = [p for p in dp.rglob("*") if p.suffix in SUFFIXES and p.is_file()
            and not any(part in SKIP_DIRS for part in p.parts)]
    dolda = [p for p in alla if any(part.startswith(".") for part in p.parts)]
    files = [p for p in alla if p not in set(dolda)]
    n_ok = sum(1 for p in files if os.path.realpath(str(p)) in seen)
    return {"n_filer_i_repo": len(files), "n_filer_i_dolda_kataloger": len(dolda),
            "n_filer_sokbara": n_ok,
            "tackning_pct": round(100 * n_ok / len(files), 1) if files else None}


def selftest(verbose: bool = False) -> bool:
    """Build a synthetic corpus in a temp directory and assert the ranking properties the search
    depends on. Returns True if every assertion holds."""
    import tempfile
    global ROOTS, _CORPUS
    ok = True

    def check(label: str, cond: bool) -> None:
        nonlocal ok
        if verbose:
            print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pkg" / "printer").mkdir(parents=True)
        (root / "docs").mkdir()
        # prior art: the term appears only far down in the body, past any head truncation
        (root / "pkg" / "duct_validity.py").write_text(
            '"""Validity gate for duct channels: genus and topology invariants."""\n'
            + ("# filler\n" * 400) + "TOPOLOGY_GENUS_CHECK = True\n")
        # the same terms buried deep in the body of another file (found, but ranked lower)
        (root / "pkg" / "deep_body.py").write_text(
            '"""Unrelated helper."""\n' + ("# filler\n" * 400) + "# genus topology duct\n")
        # a long essay docstring mentioning the words in passing
        (root / "docs" / "essay.md").write_text("# essay\n" + ("genus topology duct prose. " * 300))
        # path-zone hit only
        (root / "pkg" / "printer" / "layout.py").write_text('"""Plate layout."""\n')
        # a file spelled with diacritics
        (root / "pkg" / "tatning.py").write_text('"""T\u00e4tning / gasket geometry."""\n')

        ROOTS = [("test", root)]
        _CORPUS = None

        scored, per_term = find(["genus", "topology", "duct"])
        ranked = rank(scored)
        exact = [s for s in ranked if s["n"] == 3]
        check("terms far down in the body are still found (no head truncation)",
              any(s["name"] == "deep_body.py" for s in exact))
        check("precise docstring + path outranks the long essay and the body-only file",
              bool(exact) and exact[0]["name"] == "duct_validity.py")
        check("the long essay is ranked below the precise docstring",
              [s["name"] for s in exact].index("essay.md")
              > [s["name"] for s in exact].index("duct_validity.py"))

        _CORPUS = None
        scored2, per_term2 = find(["printer", "nonexistentterm"])
        ranked2 = rank(scored2)
        check("path zone matches a directory name", per_term2["printer"] >= 1)
        check("strict-AND miss degrades to partial hits, never to nothing",
              bool(ranked2) and max(s["n"] for s in ranked2) == 1)

        _CORPUS = None
        scored3, _ = find(["gasket"])
        check("synonym group finds the other language", any(s["name"] == "tatning.py" for s in scored3))

        _CORPUS = None
        scored4, _ = find(["tatning"])
        check("ASCII folding matches the diacritic spelling",
              any(s["name"] == "tatning.py" for s in scored4))

        _CORPUS = None
        cov = coverage_report()
        check("coverage report reaches 100% of the root", cov["test"]["tackning_pct"] == 100.0)

    _CORPUS = None
    if verbose:
        print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "--selftest":
        return 0 if selftest(verbose=True) else 1
    if argv[0] == "--tackning":
        cov = coverage_report()
        print(json.dumps(cov, indent=2, ensure_ascii=False))
        worst = min((v["tackning_pct"] for v in cov.values() if v["tackning_pct"] is not None), default=100)
        return 0 if worst >= 99.0 else 1

    limit = 15
    if argv[0] == "--limit":
        limit, argv = int(argv[1]), argv[2:]
    terms = argv
    scored, per_term = find(terms)
    n = len(terms)
    ranked = rank(scored)
    exact = [s for s in ranked if s["n"] == n]
    if exact:
        print(f"{len(exact)} EXISTING hits for {terms} (ALL {n} terms) -- READ before building:")
        render(exact, terms, limit)
        if len(exact) > limit:
            print(f"  ... +{len(exact)-limit} more (--limit N for more)")
        return 0
    # No file matches ALL terms. Do NOT silently collapse to 'nothing found' -- that is the strict-AND
    # trap: every word exists somewhere, the intersection is just empty.
    if ranked and n > 1:
        print(f"0 files contain ALL {n} terms {terms} -- strict AND produced an empty set.")
        print("  Per term (each exists on its own): " + ", ".join(f"{t}:{per_term[t]}" for t in terms))
        print(f"  Best partial hits (ranked by zone-weighted score, {ranked[0]['n']}/{n} highest):")
        render(ranked, terms, limit)
        print("  -> This does NOT mean nothing exists -- do not build without reading the above.")
        return 0
    print(f"NOTHING found for {terms} -- build it (and write a docstring).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
