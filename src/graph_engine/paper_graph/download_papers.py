#!/usr/bin/env python3
"""Bulk downloader for scientific paper metadata + abstracts.

Part of the paper-search engine: pulls metadata + abstracts for
project-relevant topics so they can later be searched / graded.

Sources
-------
* arXiv API  (https://export.arxiv.org/api/query)   -- PRIMARY, reliable.
    Atom/XML parsed with the Python stdlib (no `feedparser` dependency).
    Rate limited to ~1 request / 3 s per arXiv's guidance; paginated.
* Semantic Scholar Graph API (https://api.semanticscholar.org)  -- OPTIONAL
    enrichment for citation_count + reference_ids. Best-effort: the keyless
    shared pool is heavily rate limited (HTTP 429). When it is unavailable the
    corpus is still built from arXiv with those fields left null. Supply an API
    key via --s2-key or the SEMANTIC_SCHOLAR_API_KEY / S2_API_KEY env var for
    reliable enrichment.

Robustness
----------
* Retry with exponential backoff + jitter on network errors and 429/5xx
  (honours Retry-After).
* Skip-on-fail: a bad page / paper / enrichment call is logged and skipped,
  never aborts the run.
* Resumable checkpoint: corpus.jsonl is the dedupe source of truth and a
  <corpus>.checkpoint.json tracks per-topic pagination offsets, so a re-run
  resumes where it stopped instead of re-downloading.

Output
------
One JSON object per line (JSONL) at the --out path (default:./corpus.jsonl next
to this script). See RECORD_SCHEMA below for the field list.

Usage
-----
    python3 download_papers.py                      # default 8 topics, arXiv only
    python3 download_papers.py --per-topic 60 --page-size 25
    python3 download_papers.py --s2                 # attempt S2 enrichment
    python3 download_papers.py --topics "topology optimization" "digital twin"
    python3 download_papers.py --self-test          # offline parser unit tests

Only third-party dependency is `requests` (falls back with a clear message).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:  # pragma: no cover - environment guard
    requests = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ARXIV_API = "https://export.arxiv.org/api/query"
S2_BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "citationCount,referenceCount,externalIds,fieldsOfStudy,references.externalIds"

# arXiv Atom namespaces.
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}

USER_AGENT = "paper-graph-fetch/1.0 (research; +https://export.arxiv.org)"

DEFAULT_TOPICS = [
    "physics simulation certification",
    "digital twin uncertainty quantification",
    "optimal experimental design",
    "topology optimization",
    "parameter identifiability Fisher information",
    "model-form uncertainty",
    "conformal prediction physics",
    "sensor placement observability",
]

# One JSONL record per paper. Every key is always present (null when unknown).
RECORD_SCHEMA = {
    "id": "canonical id, e.g. 'arxiv:2607.00894'",
    "source": "'arxiv'",
    "arxiv_id": "versionless arXiv id, e.g. '2607.00894' (null if none)",
    "arxiv_id_versioned": "e.g. '2607.00894v1'",
    "doi": "DOI string (from arXiv <arxiv:doi> or S2), or null",
    "title": "paper title (whitespace-collapsed)",
    "abstract": "abstract / summary text",
    "authors": "list[str] of author names",
    "year": "int publication year, or null",
    "published": "arXiv <published> ISO timestamp",
    "updated": "arXiv <updated> ISO timestamp",
    "categories": "list[str] arXiv category terms, e.g. ['cs.CE','math.OC']",
    "primary_category": "primary arXiv category",
    "comment": "arXiv author comment (pages/figures), or null",
    "journal_ref": "journal reference, or null",
    "abs_url": "arXiv abstract page URL",
    "pdf_url": "arXiv PDF URL",
    "topic_query": "the topic string that surfaced this paper",
    "arxiv_query": "the exact arXiv search_query used",
    # --- Semantic Scholar enrichment (null unless --s2 succeeded) ---
    "citation_count": "int inbound citations, or null",
    "reference_count": "int outbound references, or null",
    "reference_ids": "list[str] reference ids ('ARXIV:..','DOI:..','S2:..')",
    "fields_of_study": "list[str] S2 fields of study",
    "s2_paper_id": "Semantic Scholar paperId, or null",
    "s2_enriched": "bool, whether S2 enrichment succeeded for this record",
    "retrieved_at": "ISO timestamp when this record was fetched",
}


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    """Timestamped log line to stderr (keeps stdout clean for the summary)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_ws(text):
    """Collapse internal whitespace; return None for empty/None."""
    if text is None:
        return None
    collapsed = " ".join(text.split())
    return collapsed or None


class RateLimiter:
    """Enforce a minimum interval between calls (arXiv asks for ~1 per 3 s)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


# --------------------------------------------------------------------------- #
# HTTP with retry / backoff
# --------------------------------------------------------------------------- #
RETRY_STATUSES = (429, 500, 502, 503, 504)


def http_request(method, url, *, params=None, json_body=None, headers=None,
                 timeout=30, max_retries=5, base_delay=3.0, label=""):
    """HTTP request with exponential backoff + jitter on network errors and
    retryable status codes. Honours the Retry-After header. Returns the final
    Response (which may still carry an error status) or raises on exhausted
    network retries."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required. Install with: pip install requests")
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(method, url, params=params, json=json_body,
                                    headers=hdrs, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1.0)
                log(f"{label} network error {type(exc).__name__}: {exc}; retry {attempt+1}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
        if resp.status_code in RETRY_STATUSES and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1.0)
            log(f"{label} HTTP {resp.status_code}; retry {attempt+1}/{max_retries} in {delay:.1f}s")
            time.sleep(delay)
            continue
        return resp
    if last_exc:  # pragma: no cover - defensive
        raise last_exc
    return resp


# --------------------------------------------------------------------------- #
# arXiv query building + parsing
# --------------------------------------------------------------------------- #
_STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with"}
_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(.+)$")
_VERSION_RE = re.compile(r"v\d+$")


def build_arxiv_query(topic: str, phrase_total: int = None, min_phrase_hits: int = 15) -> str:
    """Build an arXiv search_query for a topic.

    Strategy (adaptive):
      * If the topic already contains a quote, treat it as a user-crafted query
        and pass through as an all-fields search.
      * Otherwise try an exact phrase  all:"<topic>".  If a prior probe found
        fewer than `min_phrase_hits` results (phrase_total), fall back to an
        AND of the significant terms  all:t1 AND all:t2...  which keeps
        precision while recovering multi-concept topics that never appear
        verbatim (e.g. 'parameter identifiability Fisher information').
    """
    topic = topic.strip()
    if '"' in topic:
        return f"all:{topic}" if ":" not in topic else topic
    if phrase_total is not None and phrase_total < min_phrase_hits:
        terms = [t for t in re.split(r"[\s\-]+", topic) if t and t.lower() not in _STOPWORDS]
        if len(terms) >= 2:
            return " AND ".join(f"all:{t}" for t in terms)
    return f'all:"{topic}"'


def probe_phrase_total(topic: str, limiter: RateLimiter, cfg) -> int:
    """One cheap request to learn how many hits the exact phrase has, so we can
    decide phrase vs AND-of-terms. Returns -1 on failure (caller falls back to
    a plain phrase query)."""
    limiter.wait()
    try:
        resp = http_request("GET", ARXIV_API,
                            params={"search_query": f'all:"{topic}"', "max_results": 1},
                            timeout=cfg.timeout, max_retries=cfg.max_retries,
                            base_delay=cfg.arxiv_delay, label="[arxiv probe]")
        if resp.status_code != 200:
            return -1
        _records, total = parse_arxiv_feed(resp.text, topic, "")
        return total
    except Exception as exc:  # noqa: BLE001 - skip-on-fail
        log(f"[arxiv probe] failed for {topic!r}: {exc}")
        return -1


def parse_arxiv_feed(xml_text: str, topic_label: str, arxiv_query: str):
    """Parse an arXiv Atom feed. Returns (records, total_results).

    Robust to malformed / error entries: any entry without a valid arXiv id is
    skipped rather than raising.
    """
    root = ET.fromstring(xml_text)
    total_el = root.find("os:totalResults", NS)
    try:
        total = int(total_el.text) if total_el is not None and total_el.text else 0
    except (TypeError, ValueError):
        total = 0

    records = []
    for entry in root.findall("a:entry", NS):
        rec = _parse_entry(entry, topic_label, arxiv_query)
        if rec is not None:
            records.append(rec)
    return records, total


def _parse_entry(entry, topic_label, arxiv_query):
    raw_id = entry.findtext("a:id", default="", namespaces=NS) or ""
    m = _ARXIV_ID_RE.search(raw_id)
    if not m:  # error entry or unexpected format -> skip
        return None
    versioned = m.group(1).strip()
    arxiv_id = _VERSION_RE.sub("", versioned)

    authors = []
    for auth in entry.findall("a:author", NS):
        name = clean_ws(auth.findtext("a:name", default="", namespaces=NS))
        if name:
            authors.append(name)

    categories = [c.get("term") for c in entry.findall("a:category", NS) if c.get("term")]
    prim_el = entry.find("arxiv:primary_category", NS)
    primary = prim_el.get("term") if prim_el is not None else (categories[0] if categories else None)

    published = entry.findtext("a:published", default=None, namespaces=NS)
    year = None
    if published and len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])

    abs_url, pdf_url = None, None
    for link in entry.findall("a:link", NS):
        href, rel, title = link.get("href"), link.get("rel"), link.get("title")
        if title == "pdf" or (href and "/pdf/" in href):
            pdf_url = href
        elif rel == "alternate":
            abs_url = href
    if abs_url is None:
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    if pdf_url is None:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    return {
        "id": f"arxiv:{arxiv_id}",
        "source": "arxiv",
        "arxiv_id": arxiv_id,
        "arxiv_id_versioned": versioned,
        "doi": clean_ws(entry.findtext("arxiv:doi", default=None, namespaces=NS)),
        "title": clean_ws(entry.findtext("a:title", default="", namespaces=NS)),
        "abstract": clean_ws(entry.findtext("a:summary", default="", namespaces=NS)),
        "authors": authors,
        "year": year,
        "published": published,
        "updated": entry.findtext("a:updated", default=None, namespaces=NS),
        "categories": categories,
        "primary_category": primary,
        "comment": clean_ws(entry.findtext("arxiv:comment", default=None, namespaces=NS)),
        "journal_ref": clean_ws(entry.findtext("arxiv:journal_ref", default=None, namespaces=NS)),
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "topic_query": topic_label,
        "arxiv_query": arxiv_query,
        # S2 enrichment placeholders (null until enriched):
        "citation_count": None,
        "reference_count": None,
        "reference_ids": [],
        "fields_of_study": [],
        "s2_paper_id": None,
        "s2_enriched": False,
        "retrieved_at": iso_now(),
    }


def fetch_arxiv_page(query, start, page_size, topic_label, limiter, cfg):
    """Fetch a single page. Returns (records, total). Skip-on-fail -> ([], -1)."""
    limiter.wait()
    params = {
        "search_query": query,
        "start": start,
        "max_results": page_size,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = http_request("GET", ARXIV_API, params=params, timeout=cfg.timeout,
                            max_retries=cfg.max_retries, base_delay=cfg.arxiv_delay,
                            label=f"[arxiv {topic_label[:24]}]")
    except Exception as exc:  # noqa: BLE001
        log(f"[arxiv] request failed (start={start}): {exc}; skipping page")
        return [], -1
    if resp.status_code != 200:
        log(f"[arxiv] HTTP {resp.status_code} at start={start}; skipping page")
        return [], -1
    try:
        return parse_arxiv_feed(resp.text, topic_label, query)
    except ET.ParseError as exc:
        log(f"[arxiv] XML parse error at start={start}: {exc}; skipping page")
        return [], -1


# --------------------------------------------------------------------------- #
# Semantic Scholar enrichment (best-effort)
# --------------------------------------------------------------------------- #
def s2_enrich(records, cfg):
    """Enrich records in place with citation_count / reference_ids via the S2
    batch endpoint (one request for up to 500 ids). Returns count enriched.
    Any failure (typically 429 on the keyless pool) degrades gracefully to 0."""
    if not cfg.s2:
        return 0
    targets = [r for r in records if r.get("arxiv_id")]
    if not targets:
        return 0
    ids = [f"ARXIV:{r['arxiv_id']}" for r in targets]

    headers = {}
    if cfg.s2_key:
        headers["x-api-key"] = cfg.s2_key

    try:
        resp = http_request("POST", S2_BATCH_API, params={"fields": S2_FIELDS},
                            json_body={"ids": ids}, headers=headers, timeout=cfg.timeout,
                            max_retries=cfg.s2_max_retries, base_delay=cfg.s2_delay,
                            label="[s2]")
    except Exception as exc:  # noqa: BLE001
        log(f"[s2] enrichment request failed: {exc}; leaving citation fields null")
        return 0
    if resp.status_code != 200:
        log(f"[s2] HTTP {resp.status_code} (keyless pool is often rate-limited); "
            f"leaving citation fields null for this batch")
        return 0
    try:
        data = resp.json()
    except ValueError:
        log("[s2] non-JSON response; skipping enrichment for this batch")
        return 0

    enriched = 0
    for rec, obj in zip(targets, data):
        if not obj:
            continue
        rec["citation_count"] = obj.get("citationCount")
        rec["reference_count"] = obj.get("referenceCount")
        rec["fields_of_study"] = obj.get("fieldsOfStudy") or []
        rec["s2_paper_id"] = obj.get("paperId")
        ext = obj.get("externalIds") or {}
        if not rec.get("doi") and ext.get("DOI"):
            rec["doi"] = ext["DOI"]
        ref_ids = []
        for ref in (obj.get("references") or []):
            eids = ref.get("externalIds") or {}
            if eids.get("ArXiv"):
                ref_ids.append("ARXIV:" + eids["ArXiv"])
            elif eids.get("DOI"):
                ref_ids.append("DOI:" + eids["DOI"])
            elif ref.get("paperId"):
                ref_ids.append("S2:" + ref["paperId"])
            if len(ref_ids) >= cfg.s2_max_refs:
                break
        rec["reference_ids"] = ref_ids
        rec["s2_enriched"] = True
        enriched += 1
    return enriched


def fetch_arxiv_ids(ids, limiter, cfg, batch_size=25):
    """Fetch an explicit list of arXiv ids via the API's id_list parameter.

    Returns (records, missing_ids). Same record schema as the topic search; the
    topic_query field is set to 'id_list' so downstream stages can tell the two
    acquisition modes apart. Skip-on-fail per batch, like fetch_arxiv_page."""
    records, got = [], set()
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        limiter.wait()
        try:
            resp = http_request("GET", ARXIV_API,
                                params={"id_list": ",".join(batch), "max_results": len(batch)},
                                timeout=cfg.timeout, max_retries=cfg.max_retries,
                                base_delay=cfg.arxiv_delay, label="[arxiv id_list]")
        except Exception as exc:  # noqa: BLE001 - skip-on-fail
            log(f"[arxiv id_list] request failed for batch at {start}: {exc}; skipping batch")
            continue
        if resp.status_code != 200:
            log(f"[arxiv id_list] HTTP {resp.status_code} for batch at {start}; skipping batch")
            continue
        try:
            recs, _total = parse_arxiv_feed(resp.text, "id_list", "id_list:" + ",".join(batch))
        except ET.ParseError as exc:
            log(f"[arxiv id_list] XML parse error for batch at {start}: {exc}; skipping batch")
            continue
        for rec in recs:
            got.add(rec["arxiv_id"])
            records.append(rec)
        log(f"  id_list batch {start}..{start + len(batch) - 1}: {len(recs)} records")
    missing = [i for i in ids if i not in got]
    return records, missing


def run_ids(cfg):
    """Acquisition mode for a fixed id list: no pagination, no checkpoint topics."""
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)) or ".", exist_ok=True)
    if cfg.reset:
        for path in (cfg.out, checkpoint_path_for(cfg.out)):
            if os.path.exists(path):
                os.remove(path)
        log(f"--reset: cleared {cfg.out} and checkpoint")
    seen = load_seen_ids(cfg.out)
    limiter = RateLimiter(cfg.arxiv_delay)
    records, missing = fetch_arxiv_ids(cfg.ids, limiter, cfg)
    new_records = [r for r in records if r["id"] not in seen]
    enriched = 0
    if cfg.s2 and new_records:
        enriched = s2_enrich(new_records, cfg)
    with open(cfg.out, "a", encoding="utf-8") as fh:
        append_records(fh, new_records)
    for rec in new_records:
        seen.add(rec["id"])
    if missing:
        log(f"{len(missing)} requested ids returned no record: {missing}")
    _print_summary(cfg, {"id_list": len(new_records)}, len(new_records), enriched, len(seen))


# --------------------------------------------------------------------------- #
# Corpus + checkpoint persistence
# --------------------------------------------------------------------------- #
def load_seen_ids(corpus_path):
    """Read existing corpus (dedupe source of truth). Returns set of ids.
    Tolerates a truncated final line from an interrupted run."""
    seen = set()
    if not os.path.exists(corpus_path):
        return seen
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a partial trailing line
            if rec.get("id"):
                seen.add(rec["id"])
    return seen


def checkpoint_path_for(corpus_path):
    return corpus_path + ".checkpoint.json"


def load_checkpoint(corpus_path):
    path = checkpoint_path_for(corpus_path)
    if not os.path.exists(path):
        return {"topics": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"topics": {}}


def save_checkpoint(corpus_path, ckpt):
    ckpt["updated"] = iso_now()
    path = checkpoint_path_for(corpus_path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ckpt, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic


def append_records(fh, records):
    for rec in records:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


# --------------------------------------------------------------------------- #
# Main run loop
# --------------------------------------------------------------------------- #
def run(cfg):
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)) or ".", exist_ok=True)

    if cfg.reset:
        for p in (cfg.out, checkpoint_path_for(cfg.out)):
            if os.path.exists(p):
                os.remove(p)
        log(f"--reset: cleared {cfg.out} and checkpoint")

    seen = load_seen_ids(cfg.out)
    ckpt = load_checkpoint(cfg.out)
    ckpt.setdefault("topics", {})
    log(f"Resuming with {len(seen)} papers already in corpus; "
        f"{sum(1 for t in ckpt['topics'].values() if t.get('done'))} topics marked done")

    limiter = RateLimiter(cfg.arxiv_delay)
    grand_new = 0
    grand_enriched = 0
    per_topic_new = {}

    with open(cfg.out, "a", encoding="utf-8") as fh:
        for topic in cfg.topics:
            state = ckpt["topics"].setdefault(topic, {})
            if state.get("done"):
                log(f"topic done, skipping: {topic!r}")
                per_topic_new.setdefault(topic, 0)
                continue

            # Decide the query once per topic (adaptive phrase vs AND-of-terms).
            if not state.get("query"):
                phrase_total = probe_phrase_total(topic, limiter, cfg)
                state["query"] = build_arxiv_query(
                    topic, phrase_total=None if phrase_total < 0 else phrase_total,
                    min_phrase_hits=cfg.min_phrase_hits)
                state["phrase_total"] = phrase_total
                state.setdefault("next_start", 0)
                state.setdefault("collected", 0)
                save_checkpoint(cfg.out, ckpt)
            query = state["query"]
            log(f"topic {topic!r} -> query {query!r} (phrase_total={state.get('phrase_total')})")

            collected = state.get("collected", 0)
            topic_new = 0
            while collected < cfg.per_topic:
                start = state.get("next_start", 0)
                records, total = fetch_arxiv_page(query, start, cfg.page_size, topic, limiter, cfg)
                if total == -1:  # page failed -> stop this topic, keep checkpoint for resume
                    log(f"  page failed at start={start}; will resume {topic!r} on re-run")
                    break
                state["total"] = total
                if not records:
                    log(f"  no more results for {topic!r} (start={start}, total={total})")
                    state["done"] = True
                    break

                new_records = [r for r in records if r["id"] not in seen]
                if cfg.s2 and new_records:
                    grand_enriched += s2_enrich(new_records, cfg)

                append_records(fh, new_records)
                for r in new_records:
                    seen.add(r["id"])
                collected += len(new_records)
                topic_new += len(new_records)
                grand_new += len(new_records)
                state["collected"] = collected
                state["next_start"] = start + len(records)
                save_checkpoint(cfg.out, ckpt)
                log(f"  +{len(new_records)} new (page had {len(records)}, "
                    f"topic total collected={collected}/{cfg.per_topic}, corpus={len(seen)})")

                if cfg.max_total and grand_new >= cfg.max_total:
                    log(f"reached --max-total {cfg.max_total}; stopping")
                    per_topic_new[topic] = topic_new
                    _print_summary(cfg, per_topic_new, grand_new, grand_enriched, len(seen))
                    return
                if state["next_start"] >= total:
                    state["done"] = True
                    break

            if collected >= cfg.per_topic:
                state["done"] = True
            per_topic_new[topic] = topic_new
            save_checkpoint(cfg.out, ckpt)

    _print_summary(cfg, per_topic_new, grand_new, grand_enriched, len(seen))


def _print_summary(cfg, per_topic_new, grand_new, grand_enriched, corpus_size):
    lines = ["", "=" * 68, "DOWNLOAD SUMMARY", "=" * 68,
             f"corpus file      : {os.path.abspath(cfg.out)}",
             f"total in corpus  : {corpus_size}",
             f"new this run     : {grand_new}",
             f"S2 enriched      : {grand_enriched}" + ("" if cfg.s2 else "  (S2 disabled; pass --s2)"),
             "new per topic:"]
    for topic, n in per_topic_new.items():
        lines.append(f"    {n:4d}  {topic}")
    lines.append("=" * 68)
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Offline self-test (parser + query builder + reference extraction)
# --------------------------------------------------------------------------- #
SAMPLE_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>1077</opensearch:totalResults>
  <opensearch:itemsPerPage>2</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2607.00894v1</id>
    <title>Dynamic Adaptive Mesh Refinement for
      Topology Optimization</title>
    <updated>2026-07-01T13:00:37Z</updated>
    <published>2026-07-01T13:00:37Z</published>
    <summary>  We study adaptive mesh refinement
      applied to density-based topology optimization.  </summary>
    <link href="https://arxiv.org/abs/2607.00894v1" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2607.00894v1" rel="related" type="application/pdf" title="pdf"/>
    <arxiv:doi>10.1000/example.doi.2607</arxiv:doi>
    <arxiv:journal_ref>J. Comp. Phys. 999 (2026) 1-20</arxiv:journal_ref>
    <arxiv:comment>18 pages, 6 figures</arxiv:comment>
    <category term="cs.CE" scheme="http://arxiv.org/schemas/atom"/>
    <category term="math.OC" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.CE"/>
    <author><name>Ada Lovelace</name></author>
    <author><name>Carl Gauss</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
    <summary>bad id</summary>
  </entry>
</feed>"""

# Minimal shape of one S2 batch item, used to unit-test reference extraction.
SAMPLE_S2_OBJ = {
    "paperId": "abc123",
    "citationCount": 42,
    "referenceCount": 3,
    "externalIds": {"DOI": "10.1000/example.doi.2607", "ArXiv": "2607.00894"},
    "fieldsOfStudy": ["Engineering", "Mathematics"],
    "references": [
        {"paperId": "r1", "externalIds": {"ArXiv": "1234.5678"}},
        {"paperId": "r2", "externalIds": {"DOI": "10.9/xyz"}},
        {"paperId": "r3", "externalIds": {}},
    ],
}


class _Cfg:  # lightweight config for the self-test
    s2 = True
    s2_max_refs = 100


def self_test():
    failures = []

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            failures.append(name)

    print("1) arXiv Atom parser")
    records, total = parse_arxiv_feed(SAMPLE_XML, "topology optimization", 'all:"topology optimization"')
    check("totalResults parsed (1077)", total == 1077)
    check("error entry skipped (1 valid record)", len(records) == 1)
    r = records[0]
    check("versionless arxiv_id", r["arxiv_id"] == "2607.00894")
    check("versioned id kept", r["arxiv_id_versioned"] == "2607.00894v1")
    check("canonical id", r["id"] == "arxiv:2607.00894")
    check("title whitespace collapsed", r["title"] == "Dynamic Adaptive Mesh Refinement for Topology Optimization")
    check("abstract stripped/collapsed", r["abstract"].startswith("We study") and "  " not in r["abstract"])
    check("authors list", r["authors"] == ["Ada Lovelace", "Carl Gauss"])
    check("year from published", r["year"] == 2026)
    check("categories", r["categories"] == ["cs.CE", "math.OC"])
    check("primary_category", r["primary_category"] == "cs.CE")
    check("doi from arxiv:doi", r["doi"] == "10.1000/example.doi.2607")
    check("journal_ref", r["journal_ref"] == "J. Comp. Phys. 999 (2026) 1-20")
    check("comment", r["comment"] == "18 pages, 6 figures")
    check("pdf_url", r["pdf_url"] == "https://arxiv.org/pdf/2607.00894v1")
    check("topic_query propagated", r["topic_query"] == "topology optimization")
    check("s2 placeholders null", r["citation_count"] is None and r["s2_enriched"] is False)

    print("2) adaptive query builder")
    check("common phrase kept quoted",
          build_arxiv_query("topology optimization", phrase_total=1077) == 'all:"topology optimization"')
    check("sparse phrase -> AND of terms",
          build_arxiv_query("digital twin uncertainty quantification", phrase_total=0)
          == "all:digital AND all:twin AND all:uncertainty AND all:quantification")
    check("hyphen split + stopword drop",
          build_arxiv_query("model-form uncertainty of the system", phrase_total=0)
          == "all:model AND all:form AND all:uncertainty AND all:system")
    check("unknown phrase_total defaults to phrase",
          build_arxiv_query("sensor placement observability") == 'all:"sensor placement observability"')

    print("3) S2 reference-id extraction")
    rec = dict(records[0])
    rec["reference_ids"] = []
    # exercise the extraction block directly against a sample S2 object
    obj = SAMPLE_S2_OBJ
    ref_ids = []
    for ref in obj["references"]:
        eids = ref.get("externalIds") or {}
        if eids.get("ArXiv"):
            ref_ids.append("ARXIV:" + eids["ArXiv"])
        elif eids.get("DOI"):
            ref_ids.append("DOI:" + eids["DOI"])
        elif ref.get("paperId"):
            ref_ids.append("S2:" + ref["paperId"])
    check("reference ids mapped by priority",
          ref_ids == ["ARXIV:1234.5678", "DOI:10.9/xyz", "S2:r3"])
    check("citationCount readable", obj["citationCount"] == 42)

    print("4) JSONL round-trip")
    line = json.dumps(records[0], ensure_ascii=False)
    back = json.loads(line)
    check("record survives json round-trip", back == records[0])
    check("all schema keys present", set(records[0].keys()) == set(RECORD_SCHEMA.keys()))

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("SELF-TEST PASSED (all checks green)")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv):
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.jsonl")
    p = argparse.ArgumentParser(description="Bulk-download paper metadata + abstracts (arXiv + optional Semantic Scholar).")
    p.add_argument("--topics", nargs="+", default=None,
                   help="Topic query strings (default: 8 built-in topics).")
    p.add_argument("--topics-file", default=None,
                   help="File with one topic per line (overrides --topics).")
    p.add_argument("--ids-file", default=None,
                   help="File with one arXiv id per line ('#' comments allowed). Fetches exactly "
                        "those ids via the API's id_list parameter instead of running topic searches.")
    p.add_argument("--out", default=default_out, help=f"Output JSONL path (default: {default_out}).")
    p.add_argument("--per-topic", type=int, default=50, help="Target new papers per topic (default 50).")
    p.add_argument("--page-size", type=int, default=100, help="arXiv results per request (default 100, arXiv max ~2000).")
    p.add_argument("--max-total", type=int, default=0, help="Global cap on new papers this run (0 = unlimited).")
    p.add_argument("--arxiv-delay", type=float, default=3.0, help="Min seconds between arXiv requests (default 3.0).")
    p.add_argument("--min-phrase-hits", type=int, default=15,
                   help="Below this many exact-phrase hits, fall back to AND-of-terms (default 15).")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout seconds.")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries for arXiv requests.")
    # Semantic Scholar
    p.add_argument("--s2", dest="s2", action="store_true", default=False,
                   help="Attempt Semantic Scholar enrichment (citations + references).")
    p.add_argument("--no-s2", dest="s2", action="store_false", help="Disable S2 enrichment (default).")
    p.add_argument("--s2-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY"),
                   help="Semantic Scholar API key (or set SEMANTIC_SCHOLAR_API_KEY / S2_API_KEY).")
    p.add_argument("--s2-delay", type=float, default=3.0, help="Base backoff seconds for S2 (default 3.0).")
    p.add_argument("--s2-max-retries", type=int, default=3, help="Max retries for S2 (default 3).")
    p.add_argument("--s2-max-refs", type=int, default=100, help="Cap reference_ids stored per paper (default 100).")
    # control
    p.add_argument("--reset", action="store_true", help="Delete existing corpus + checkpoint and start fresh.")
    p.add_argument("--self-test", action="store_true", help="Run offline unit tests (parser/query/refs) and exit.")
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    if cfg.self_test:
        return self_test()
    if requests is None:
        print("ERROR: the 'requests' package is required (pip install requests).", file=sys.stderr)
        return 2
    cfg.ids = []
    if cfg.ids_file:
        with open(cfg.ids_file, "r", encoding="utf-8") as fh:
            cfg.ids = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        cfg.topics = []
        log(f"id list ({len(cfg.ids)}) from {cfg.ids_file}")
        log(f"out={cfg.out} arxiv_delay={cfg.arxiv_delay}s s2={cfg.s2}")
        try:
            run_ids(cfg)
        except KeyboardInterrupt:
            log("interrupted")
            return 130
        return 0
    if cfg.topics_file:
        with open(cfg.topics_file, "r", encoding="utf-8") as fh:
            cfg.topics = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    elif cfg.topics is None:
        cfg.topics = list(DEFAULT_TOPICS)
    log(f"topics ({len(cfg.topics)}): {cfg.topics}")
    log(f"out={cfg.out} per_topic={cfg.per_topic} page_size={cfg.page_size} "
        f"arxiv_delay={cfg.arxiv_delay}s s2={cfg.s2}")
    try:
        run(cfg)
    except KeyboardInterrupt:
        log("interrupted; checkpoint saved, re-run to resume")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
