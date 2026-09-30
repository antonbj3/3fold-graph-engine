"""Operate this research graph with the pinned existing Graph Engine API.

Usage: python graphctl.py validate|rank|record [options]. Writes stay in this job.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/anton/research/inference_training_20260921/jobs/research_graph_pilot')
from build_pilot import load_tools
ROOT = Path(__file__).resolve().parent / 'state'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_state() -> tuple[dict, list[dict]]:
    graph = json.loads((ROOT / "GRAPH.json").read_text())
    ledger = [json.loads(s) for s in (ROOT / "LEDGER.jsonl").read_text().splitlines()]
    return graph, ledger


def ref_health(ledger_path: Path) -> dict:
    refs = json.loads((ROOT / "SOURCE_REFERENCES.json").read_text())
    stale = []
    corrupt = []
    unavailable = []
    frozen_ref_count = 0
    for key, ref in refs.items():
        p = Path(ref["path"])
        original_valid = p.is_file() and digest(p) == ref["sha256"]
        if not original_valid:
            stale.append(key)
        if ref.get("frozen_copy"):
            frozen_ref_count += 1
            snapshot = ROOT / ref["frozen_copy"]
            if not snapshot.is_file() or digest(snapshot) != ref["sha256"]:
                corrupt.append(key)
                if not original_valid:
                    unavailable.append(key)
        elif not original_valid:
            unavailable.append(key)
    frozen = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    # A missing/corrupt frozen reference is distinct from drift in its original.
    for key, ref in frozen.items():
        p = ROOT / ref["snapshot"]
        if not p.is_file() or digest(p) != ref["sha256"]:
            corrupt.append(key)
            unavailable.append(key)
    recorded = 0
    binding_errors = []
    all_sources = {**frozen, **refs}
    for line in ledger_path.read_text().splitlines():
        row = json.loads(line)
        for key in row.get("source_keys", []):
            if key not in all_sources or row.get("source_sha256", {}).get(key) != all_sources[key]["sha256"]:
                binding_errors.append({"fold": row.get("id"), "source_key": key})
        if "artifact" not in row:
            continue
        recorded += 1
        artifact = row["artifact"]
        original = Path(artifact["path"])
        original_valid = original.is_file() and digest(original) == artifact["sha256"]
        if not original_valid:
            stale.append(row["id"])
        snapshot_rel = row.get("artifact_snapshot")
        snapshot = ROOT / snapshot_rel if snapshot_rel else None
        snapshot_valid = bool(snapshot and snapshot.is_file() and digest(snapshot) == artifact["sha256"])
        if not snapshot_valid:
            corrupt.append(row["id"])
            if not original_valid:
                unavailable.append(row["id"])
    return {"referenced_files": len(refs), "source_drift": stale,
            "frozen_reference_copies": frozen_ref_count,
            "frozen_snapshots": len(frozen), "recorded_artifacts": recorded,
            "corrupt_snapshots": corrupt, "unavailable_evidence": unavailable,
            "ledger_source_binding_errors": binding_errors}


def compute(graph: dict, ledger_path: Path) -> dict:
    tool = load_tools()
    validation = tool.validate(graph, ledger_path=ledger_path, repo_root=ROOT)
    health = ref_health(ledger_path)
    refresh = tool.refresh_status(graph, ledger_path=ledger_path)
    validation["source_health"] = health
    validation["refresh_status"] = refresh
    validation["ok"] = (validation["ok"] and not health["corrupt_snapshots"]
                        and not health["unavailable_evidence"] and not health["ledger_source_binding_errors"])
    # Do not expose a feasible frontier from an integrity-invalid evidence graph.
    return {"validation": validation,
            "next_actions": tool.next_actions(graph) if validation["ok"] else [],
            "priority": tool.priority(graph) if validation["ok"] else []}


def write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=ROOT,
                                     prefix=".graphctl-", delete=False) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, path)


def save_rank(out: dict) -> None:
    for name, value in (("VALIDATION.json", out["validation"]),
                        ("NEXT_ACTIONS.json", out["next_actions"]),
                        ("PRIORITY.json", out["priority"])):
        write_atomic(ROOT / name, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def command_rank(validate_only: bool) -> None:
    with writer_lock():
        _rank_locked(validate_only)


def _rank_locked(validate_only: bool) -> None:
    graph, _ = read_state()
    out = compute(graph, ROOT / "LEDGER.jsonl")
    save_rank(out)
    valid = out["validation"]
    top = next((r for r in out["priority"] if r["actionable"]), None)
    print(json.dumps({"ok": valid["ok"], "nodes": valid["node_count"],
                      "errors": valid["errors"], "warnings": valid["warnings"],
                      "source_health": valid["source_health"],
                      "stale_evidence": valid["refresh_status"]["flags"],
                      "top_action": None if validate_only or top is None else top},
                     indent=2, ensure_ascii=False))
    if not valid["ok"]:
        raise SystemExit(2)


@contextlib.contextmanager
def writer_lock():
    with (ROOT / ".graphctl.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another graph writer holds .graphctl.lock")
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def command_record(args: argparse.Namespace) -> None:
    with writer_lock():
        _record_locked(args)


def _record_locked(args: argparse.Namespace) -> None:
    graph, ledger = read_state()
    byid = {n["id"]: n for n in graph["nodes"]}
    if args.node not in byid:
        raise SystemExit(f"unknown node {args.node}")
    artifact = args.artifact.resolve(strict=True)
    if not artifact.is_file() or not stat.S_ISREG(artifact.stat().st_mode):
        raise SystemExit("artifact must be a regular file")
    if artifact.stat().st_size > 16 * 1024 * 1024:
        raise SystemExit("recorded artifact exceeds 16 MiB; record a small immutable review/manifest")
    if artifact in {ROOT / "GRAPH.json", ROOT / "LEDGER.jsonl"}:
        raise SystemExit("cannot record mutable graph/ledger as evidence")
    if args.status and (not args.verdict_reason or not args.scope or args.evidence_kind == "artifact_only"):
        raise SystemExit("status change requires --verdict-reason, --scope, and adjudicating --evidence-kind")
    if args.status in ("PROVEN", "REFUTED") and artifact.name.lower() in {
            "next_experiment.md", "scheduler_test_prereg.md", "scoring.md", "prereg.md"}:
        raise SystemExit("a plan/preregistration cannot adjudicate a result")
    if args.status == "PROVEN" and byid[args.node]["type"] == "EMPIRICAL" and args.evidence_kind != "measured_result":
        raise SystemExit("empirical PROVEN requires measured_result evidence")
    if args.gate_open and not args.gate_reason:
        raise SystemExit("opening a regime gate requires --gate-reason")
    if args.status == "PROVEN":
        unmet = [d for d in byid[args.node]["depends_on"] if byid[d]["status"] != "PROVEN"]
        if unmet:
            raise SystemExit(f"cannot prove node with unmet dependencies: {unmet}")
    if args.status in ("PROVEN", "REFUTED") or args.gate_open:
        # Execute the strengthened EXISTING fold gate before snapshots/status mutation.
        # A saved green report is never sufficient; fresh probes run every round.
        submission_path = getattr(args, "admission_submission", None)
        template_path = getattr(args, "admission_template", None)
        registry_path = getattr(args, "admission_registry", None)
        if not submission_path or not template_path or not registry_path:
            raise SystemExit("adjudication/gate opening requires --admission-submission, --admission-template and --admission-registry")
        submission = json.loads(Path(submission_path).read_text())
        if submission.get("node") != args.node:
            raise SystemExit("admission submission must bind the recorded node")
        receipt_artifact = Path(submission.get("rerun", {}).get("artifact", ""))
        if not receipt_artifact.is_absolute():
            receipt_artifact = Path(submission_path).resolve().parent / receipt_artifact
        if receipt_artifact.resolve() != artifact:
            raise SystemExit("admission receipt must cite the exact recorded artifact")
        from graph_engine.tools.fold_gate_v2 import fold_gate_v2_round
        with tempfile.TemporaryDirectory(prefix="admission-round-", dir=ROOT) as temp:
            report = fold_gate_v2_round([submission], json.loads(Path(template_path).read_text()), temp,
                       verification_registry=json.loads(Path(registry_path).read_text()))
        if report["accepted_indices"] != [0]:
            raise SystemExit(json.dumps({"admission_blocked": True, "round": report}, default=str))
    if args.gate_open and "regime_gate" not in byid[args.node]:
        raise SystemExit("node has no regime gate")
    seq = 1 + max([int(r["id"].split("-")[-1]) for r in ledger
                   if r.get("id", "").startswith("F-UPDATE-")] or [0])
    fid = f"F-UPDATE-{seq:04d}"
    reference = {"path": str(artifact), "sha256": digest(artifact),
                 "bytes": artifact.stat().st_size}
    if args.dry_run:
        dry_snapshot_dir = Path(tempfile.mkdtemp(prefix=".graphctl-dry-", dir=ROOT))
        snapshot = dry_snapshot_dir / f"{fid}{artifact.suffix}"
    else:
        dry_snapshot_dir = None
        snapshot = ROOT / "recorded" / f"{fid}{artifact.suffix}"
        snapshot.parent.mkdir(exist_ok=True)
        if snapshot.exists():
            raise SystemExit(f"snapshot already exists: {snapshot}")
    shutil.copyfile(artifact, snapshot)
    if digest(snapshot) != reference["sha256"]:
        snapshot.unlink(missing_ok=True)
        raise SystemExit("artifact changed during snapshot copy")
    ledger.append({"id": fid, "ts": datetime.now(timezone.utc).isoformat(),
                   "node": args.node, "claim": args.claim,
                   "verdict": "RECORDED_UNADJUDICATED" if not args.status else args.status,
                   "evidence_kind": args.evidence_kind,
                   "verdict_reason": args.verdict_reason, "artifact": reference,
                   "artifact_snapshot": str(snapshot.relative_to(ROOT)),
                   "supersedes": None})
    byid[args.node]["evidence"].append(fid)
    if args.status:
        byid[args.node]["status"] = args.status
        byid[args.node]["adjudication_scope"] = args.scope
        byid[args.node]["adjudication_reason"] = args.verdict_reason
    if args.gate_open:
        byid[args.node]["regime_gate"] = {"open": True, "blocked_by": "", "opened_reason": args.gate_reason}
    pending = ROOT / ".graphctl-pending-ledger.jsonl"
    try:
        pending.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ledger))
        out = compute(graph, pending)
        if not out["validation"]["ok"]:
            raise SystemExit(json.dumps(out["validation"]["errors"], indent=2))
    except BaseException:
        if not args.dry_run:
            snapshot.unlink(missing_ok=True)
        raise
    finally:
        pending.unlink(missing_ok=True)
        if args.dry_run and dry_snapshot_dir is not None:
            shutil.rmtree(dry_snapshot_dir)
    preview = {"fold": fid, "node": args.node, "new_status": byid[args.node]["status"],
               "artifact": reference, "artifact_snapshot": ledger[-1]["artifact_snapshot"],
               "top_action": next((r["id"] for r in out["priority"] if r["actionable"]), None),
               "validation_ok": True, "dry_run": args.dry_run}
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    history = ROOT / "history"
    history.mkdir(exist_ok=True)
    for name in ("GRAPH.json", "LEDGER.jsonl", "VALIDATION.json", "NEXT_ACTIONS.json", "PRIORITY.json"):
        shutil.copy2(ROOT / name, history / f"{fid}-{name}")
    write_atomic(ROOT / "LEDGER.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ledger))
    write_atomic(ROOT / "GRAPH.json", json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
    save_rank(out)
    with (ROOT / "UPDATE_LOG.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(preview, ensure_ascii=False) + "\n")
    print(json.dumps(preview, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("validate")
    subs.add_parser("rank")
    rec = subs.add_parser("record")
    rec.add_argument("--node", required=True)
    rec.add_argument("--artifact", required=True, type=Path)
    rec.add_argument("--claim", required=True, help="Scoped observed result, not an inferred mechanism")
    rec.add_argument("--status", choices=("OPEN", "PROVEN", "REFUTED"))
    rec.add_argument("--scope")
    rec.add_argument("--evidence-kind", choices=("artifact_only", "measured_result", "exact_proof", "source_audit", "evaluation"), default="artifact_only")
    rec.add_argument("--verdict-reason")
    rec.add_argument("--gate-open", action="store_true")
    rec.add_argument("--gate-reason")
    rec.add_argument("--dry-run", action="store_true")
    rec.add_argument("--admission-submission", type=Path)
    rec.add_argument("--admission-template", type=Path)
    rec.add_argument("--admission-registry", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        command_record(args)
    else:
        command_rank(args.command == "validate")


if __name__ == "__main__":
    main()
