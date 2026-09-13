"""Offline pytest wrapper for the paper_graph pipeline.

The shipped metadata cache (examples/paper_graph/corpus.jsonl, 50 arXiv records: titles,
abstracts, authors, categories) is the input, so the default tests need no network. The
fetch stage itself is network-dependent and is skipped unless PAPER_GRAPH_NETWORK=1.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "paper_graph"
CORPUS = EXAMPLE_DIR / "corpus.jsonl"
IDS = EXAMPLE_DIR / "arxiv_ids.txt"

sys.path.insert(0, str(REPO_ROOT / "src"))

pytest.importorskip("sklearn")
pytest.importorskip("networkx")
pytest.importorskip("scipy")

from graph_engine.paper_graph import build_paper_graph as bpg  # noqa: E402
from graph_engine.paper_graph import download_papers as dl  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_paper_graph_example", EXAMPLE_DIR / "run_paper_graph_example.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shipped_id_list_and_corpus_agree():
    ids = [ln.strip() for ln in IDS.read_text().splitlines()
           if ln.strip() and not ln.startswith("#")]
    corpus = [json.loads(ln) for ln in CORPUS.read_text().splitlines() if ln.strip()]
    assert len(ids) == 50
    assert len(corpus) == len(ids)
    assert {r["arxiv_id"] for r in corpus} == set(ids)
    assert all(r["abstract"] and r["title"] and r["authors"] for r in corpus)


def test_download_papers_self_test():
    assert dl.self_test() == 0


def test_build_and_search_on_shipped_corpus(tmp_path):
    out_prefix = str(tmp_path / "paper_graph")
    rc = bpg.main(["--corpus", str(CORPUS), "--out", out_prefix, "--no-graphml"])
    assert rc == 0
    stats = json.loads(Path(out_prefix + "_stats.json").read_text())
    assert stats["n_nodes"] == 50
    assert stats["n_edges"] > 0

    runner = _load_runner()
    S = runner.load_search_graph(out_prefix + ".search.pkl")
    G = runner.overlay_knowledge(S)
    assert [n for n, d in G.nodes(data=True) if d.get("kind") == "knowledge"]
    result = runner.evaluate(G)
    assert result["n_papers"] == 50
    assert len(result["ranking"]) == 50
    assert result["n_knowledge_nodes"] >= 2
    # the ranking must be defined for every paper and carry the baseline columns
    assert all(set(row) >= {"paper", "struct", "cite", "semantic", "cls"}
               for row in result["ranking"])


def test_shipped_search_eval_matches_a_fresh_run(tmp_path):
    shipped = json.loads((EXAMPLE_DIR / "paper_graph_search_eval.json").read_text())
    out_prefix = str(tmp_path / "paper_graph")
    assert bpg.main(["--corpus", str(CORPUS), "--out", out_prefix, "--no-graphml"]) == 0
    runner = _load_runner()
    fresh = runner.evaluate(runner.overlay_knowledge(
        runner.load_search_graph(out_prefix + ".search.pkl")))
    for key in ("n_papers", "n_knowledge_nodes", "n_missing_knowledge_pairs",
                "classes", "gate"):
        assert fresh[key] == shipped[key]
    assert fresh["rho_struct_semantic"] == pytest.approx(shipped["rho_struct_semantic"], rel=1e-6)


@pytest.mark.skipif(os.environ.get("PAPER_GRAPH_NETWORK") != "1",
                    reason="network fetch; set PAPER_GRAPH_NETWORK=1 to run")
def test_fetch_id_list_from_arxiv(tmp_path):
    out = tmp_path / "corpus.jsonl"
    rc = dl.main(["--ids-file", str(IDS), "--out", str(out)])
    assert rc == 0
    records = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert len(records) == 50
