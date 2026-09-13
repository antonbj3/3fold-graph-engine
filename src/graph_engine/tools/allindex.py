#!/usr/bin/env python3
"""allindex.py -- recursive SQLite FTS5 full-text index over one or more source trees.

Motivation: a zone-weighted search tool only covers the roots it is configured with; trees that
are not in that list are structurally invisible to it whatever the query. This builder indexes
whole trees so the search surface is a strict superset of the narrower tool's coverage.

Roots come from $ALLINDEX_ROOTS as `label=path` entries separated by `:` (default: `cwd=.`).
A file that lies under two roots gets both labels (comma list in `roots`) and is counted once
(path is the primary key).

Per file: {path, roots, name, ext, mtime, size, docstring (first docstring/heading line), head
(leading text -- only for text-like extensions; binaries get an empty head but are still ROWED so
the file count is usable as blind-spot evidence)}.

Index/stat locations come from $ALLINDEX_DB and $ALLINDEX_STATS (defaults under ./data/allindex/).

CLI:
  python3 allindex.py build          # (re)build the whole index, write the stats JSON
                                              # (build time, files per root, files excluded per reason)
  python3 allindex.py find <word...> # search an EXISTING index (build first). FTS5 MATCH with
 # AND semantics, plus the AND-collapse antidote: if no file
                                              # matches ALL terms but some term matches somewhere, print
                                              # per-term counts and the best partial hit instead of
                                              # silently collapsing to "nothing".
  python3 allindex.py stats          # print the latest build stats
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = os.environ.get("ALLINDEX_DB", os.path.join("data", "allindex", "allindex.db"))
STATS_PATH = os.environ.get("ALLINDEX_STATS", os.path.join("data", "allindex", "allindex_stats.json"))


def _roots_from_env() -> list[tuple[str, str]]:
    """(label, path) pairs from $ALLINDEX_ROOTS (`label=path` entries separated by `:`).
    Order matters only for which root "discovers" a file first when labels overlap; a nested root
    still records its own label."""
    spec = os.environ.get("ALLINDEX_ROOTS", "").strip()
    if not spec:
        return [("cwd", os.getcwd())]
    out = []
    for item in spec.split(":"):
        if not item.strip():
            continue
        label, _, path = item.partition("=")
        out.append((label.strip() or "root", os.path.expanduser((path or label).strip())))
    return out


ROOTS = _roots_from_env()

# directory *names* pruned everywhere they occur (binary/noise, never authored content worth a
# text read -- MEASURED per-root in stats.excluded_dirs, not silently dropped)
EXCLUDE_DIR_NAMES = {
    "__pycache__", ".git", "node_modules", "browser_profile",
}
EXCLUDE_DIR_PREFIXES = (".venv",)

TEXT_EXTS = {
    ".py", ".md", ".json", ".jsonl", ".txt", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".rst", ".html", ".htm", ".csv", ".log", ".svg", ".xml",
}
MAX_READ_FILE_SIZE = 50 * 1024 * 1024  # don't even open text files bigger than this for a head-read
HEAD_BYTES = 2200          # DATA files (.json/.jsonl/.csv/.log/...): only a head is indexed

# Code and documents are indexed IN FULL. Measured on one tree: with a flat 2200 B cap, 80.2 of
# 3519.6 MB (2.3 %) of the content was indexed and 32 222 of 47 986 text files (67.1 %) were only
# partly searchable. `head` is an INDEXED fts5 column, so truncation is a search gap, not just a
# display choice. The mass sits in data (.jsonl 2441 MB, .json 459 MB), not in the source: .py
# 174.1 MB + .md 56.7 MB = 231 MB. Hence: full index for code/documents, capped head for data.
KOD_EXTS = {".py", ".md", ".sh", ".rst", ".toml", ".ini", ".cfg", ".yaml", ".yml"}
KOD_HEAD_BYTES = 1024 * 1024   # effectively the whole file; guards only pathological cases


def _head_cap(ext: str) -> int:
    return KOD_HEAD_BYTES if ext in KOD_EXTS else HEAD_BYTES

_PY_DOCSTRING_RE = re.compile(r'"""(.+?)(?:\n|""")', re.S)


def _is_excluded_dir(name: str) -> bool:
    return name in EXCLUDE_DIR_NAMES or any(name.startswith(p) for p in EXCLUDE_DIR_PREFIXES)


def head_and_doc(path: str, ext: str) -> tuple[str, str]:
    """Read first HEAD_BYTES of a text-like file; derive a one-line docstring/heading from it.
    Mirrors tool_find.doc_line's convention (py: triple-quote docstring; md: first heading/
    non-frontmatter line; other text: first non-empty line) so allindex results read consistently
    with tool_find results."""
    if ext not in TEXT_EXTS:
        return "", ""
    try:
        if os.path.getsize(path) > MAX_READ_FILE_SIZE:
            return "", ""
        with open(path, "r", errors="ignore") as f:
            head = f.read(_head_cap(ext))
    except Exception:
        return "", ""
    # By convention the docstring/heading sits at the START of the file, so look only there
    # however much of the content is indexed. Keeps the display line (short) separate from the
 # search surface (the whole file).
    top = head[:HEAD_BYTES]
    if ext == ".py":
        m = _PY_DOCSTRING_RE.search(top)
        doc = (m.group(1).strip() if m else "")[:200]
    elif ext == ".md":
        doc = ""
        for line in top.splitlines():
            s = line.strip()
            if s and not s.startswith(("---", "name:", "description:", "metadata:", "type:")):
                doc = s.lstrip("# ").strip()[:200]
                break
    else:
        doc = next((ln.strip()[:200] for ln in top.splitlines() if ln.strip()), "")
    return head, doc


# ------------------------------------------------------------------------------------------- build
def build(verbose: bool = True) -> dict:
    t0 = time.time()
    files: dict[str, dict] = {}
    per_root_new = {tag: 0 for tag, _ in ROOTS}
    per_root_seen = {tag: 0 for tag, _ in ROOTS}  # incl. already-tagged-by-another-root
    excluded_dirs_count = {tag: 0 for tag, _ in ROOTS}
    missing_roots = []

    for tag, root in ROOTS:
        if not os.path.isdir(root):
            missing_roots.append({"tag": tag, "path": root})
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            keep = []
            for d in dirnames:
                if _is_excluded_dir(d):
                    excluded_dirs_count[tag] += 1
                else:
                    keep.append(d)
            dirnames[:] = keep
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                per_root_seen[tag] += 1
                if p in files:
                    if tag not in files[p]["roots"]:
                        files[p]["roots"].append(tag)
                    continue
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                ext = os.path.splitext(fn)[1].lower()
                head, doc = head_and_doc(p, ext)
                files[p] = {
                    "path": p, "roots": [tag], "name": fn, "ext": ext,
                    "mtime": st.st_mtime, "size": st.st_size, "docstring": doc, "head": head,
                }
                per_root_new[tag] += 1

    build_scan_s = time.time() - t0

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE VIRTUAL TABLE idx USING fts5(path UNINDEXED, roots UNINDEXED, name, docstring, "
        "head, ext UNINDEXED, mtime UNINDEXED, size UNINDEXED)"
    )
    rows = [
        (r["path"], ",".join(r["roots"]), r["name"], r["docstring"], r["head"], r["ext"],
         r["mtime"], r["size"])
        for r in files.values()
    ]
    conn.executemany("INSERT INTO idx (path, roots, name, docstring, head, ext, mtime, size) "
                      "VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    total_s = time.time() - t0
    stats = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_files_indexed_total": len(files),
        "n_files_per_root_new": per_root_new,       # files whose CONTENT was read under this root
        "n_files_per_root_seen": per_root_seen,      # incl. re-hits of already-known paths (overlap)
        "excluded_dirs_pruned_per_root": excluded_dirs_count,
        "missing_roots": missing_roots,
        "db_path": DB_PATH,
        "db_size_bytes": os.path.getsize(DB_PATH),
        "scan_seconds": round(build_scan_s, 3),
        "total_build_seconds": round(total_s, 3),
    }
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    if verbose:
        print(f"allindex build: {len(files)} files indexed in {total_s:.2f}s "
              f"-> {DB_PATH} ({stats['db_size_bytes']/1e6:.1f} MB)")
        for tag, _ in ROOTS:
            print(f"  [{tag}] {per_root_new[tag]} files "
                  f"({'MISSING ROOT' if any(m['tag']==tag for m in missing_roots) else 'ok'})")
    return stats


# -------------------------------------------------------------------------------------------- find
_FTS_SPECIAL = re.compile(r'[^A-Za-z0-9_.]')


def _fts_token(term: str) -> str:
    """Quote a raw search term as a single FTS5 phrase token (handles '-', '/', etc. safely)."""
    return '"' + term.replace('"', '""') + '"'


def _query(conn: sqlite3.Connection, match_expr: str, limit: int = 500):
    cur = conn.execute(
        "SELECT path, roots, name, docstring, mtime FROM idx WHERE idx MATCH ? "
        "ORDER BY rank LIMIT ?", (match_expr, limit))
    return cur.fetchall()


def find(terms: list[str], limit: int = 25) -> dict:
    if not os.path.exists(DB_PATH):
        return {"error": f"no index at {DB_PATH} -- run `python3 scripts/allindex.py build` first"}
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    and_expr = " ".join(_fts_token(t) for t in terms)
    exact = _query(conn, and_expr, limit=10000)
    per_term = {}
    partial_best = []
    if not exact and len(terms) > 1:
        for t in terms:
            per_term[t] = len(_query(conn, _fts_token(t), limit=10000))
        # best partial: files matching the most terms (OR all terms, count per-file term overlap)
        or_expr = " OR ".join(_fts_token(t) for t in terms)
        or_rows = _query(conn, or_expr, limit=2000)
        scored = []
        for path, roots, name, doc, mtime in or_rows:
            hay = (name + " " + (doc or "")).lower()
            n_matched = sum(1 for t in terms if t.lower() in hay)
            scored.append((n_matched, path, roots, name, doc))
        scored.sort(key=lambda s: -s[0])
        partial_best = scored[:15]
    conn.close()
    elapsed = time.time() - t0
    return {
        "terms": terms, "n_exact": len(exact), "exact": exact[:limit],
        "per_term": per_term, "partial_best": partial_best,
        "search_seconds": round(elapsed, 4),
    }


def _print_find(res: dict):
    if "error" in res:
        print(res["error"])
        return
    terms = res["terms"]
    if res["n_exact"]:
        print(f"★{res['n_exact']} EXISTERANDE traffar for {terms} (ALLA {len(terms)} termer, "
              f"{res['search_seconds']}s) -- LAS innan du bygger:")
        for path, roots, name, doc, mtime in res["exact"][:25]:
            print(f"  [{roots}] {path} -- {doc}")
        if res["n_exact"] > 25:
            print(f"  ... +{res['n_exact']-25} till")
        return
    if res["partial_best"] and len(terms) > 1:
        print(f"warning: 0 files contain ALL {len(terms)} terms {terms} ({res['search_seconds']}s) "
              "-- strict AND may have intersected down to an empty set.")
        print("  Per-term: " + ", ".join(f"{t}:{res['per_term'].get(t,0)}" for t in terms))
        print(f"  Best partial matches:")
        for n_matched, path, roots, name, doc in res["partial_best"][:15]:
            print(f"  [{n_matched}/{len(terms)}] [{roots}] {path} -- {doc}")
        return
    print(f"NOTHING found for {terms} ({res['search_seconds']}s) -- build it (and write a docstring).")


# ---------------------------------------------------------------------------------- name-similarity
# Shared by cell_registry_v1's dup-marking AND drc.py's DRC6 -- ONE implementation, not two (avoiding
# exactly the kind of silent-duplicate this whole tool exists to catch).
def normalize_cell_name(stem: str) -> str:
    """Strip version/variant suffixes so 'feem_b1_master_v5_2' and 'feem_b1_master_v5_3' compare
    close to their shared root 'feem_b1_master'."""
    s = stem.lower()
    s = re.sub(r'(_v\d+)+$', '', s)
    s = re.sub(r'_(final|finalize|gen|cell|pipeline|run|v0|v1|v2)$', '', s)
    return s


def similar_name(a: str, b: str, threshold: float = 0.8) -> tuple[bool, float]:
    """~80% name similarity (difflib ratio over normalized stems) used by the duplicate guard."""
    import difflib
    na, nb = normalize_cell_name(a), normalize_cell_name(b)
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return ratio >= threshold, round(ratio, 4)


# ----------------------------------------------------------------------------------------------CLI
def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "build":
        build()
        return 0
    if cmd == "find":
        terms = sys.argv[2:]
        if not terms:
            print("usage: allindex.py find <term...>")
            return 2
        res = find(terms)
        _print_find(res)
        return 0 if (res.get("n_exact") or res.get("error")) else 1
    if cmd == "stats":
        if not os.path.exists(STATS_PATH):
            print(f"no stats yet at {STATS_PATH} -- run `build` first")
            return 1
        print(open(STATS_PATH).read())
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
