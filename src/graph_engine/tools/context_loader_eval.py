#!/usr/bin/env python3
"""
context_loader_eval.py -- the FROZEN head-to-head eval described in
context_loader_PREREG.json. Run ONCE after the PREREG is locked; do not
re-tune context_loader.py's parameters against these numbers afterward.

Computes, for every task (dev-set t1-t5/t2b + held-out h1-h3) and every
method (naive-grep, BM25 field-boosted, TF-IDF-cosine field-boosted, LSA
r=chosen-by-spectrum): the full rank of gold_primary, precision@1/3/5
against {gold_primary} U gold_acceptable, and a hit@k boolean.

Then tallies the PRE-REGISTERED headline comparison (TF-IDF-cosine vs BM25,
hit@3) on the held-out set alone (the blind test) and separately on the
dev set (disclosed non-blind, sanity-check only), and computes a small-n
Beta(1,1) posterior for P(TF-IDF-cosine beats BM25 | this corpus).
"""
import json
import os

from context_loader import ContextIndex, DEFAULT_MEMDIR

PREREG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_loader_PREREG.json")
EVIDENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_loader_evidence.json")

METHODS = ["grep", "bm25", "tfidf", "lsa"]
METHODS_ALL = METHODS + ["fused"]  # 'fused' (RRF) is POST-HOC, added after diagnosing the h1 length pathology
KS = [1, 3, 5]


def eval_task(idx: ContextIndex, task: dict, method: str):
    gold_primary = task["gold_primary"]
    relevant = set([gold_primary] + task.get("gold_acceptable", []))
    full = idx.rank_full(task["text"], method=method)
    names = [n for n, s in full]
    rank_primary = names.index(gold_primary) + 1 if gold_primary in names else None
    out = {"rank_primary": rank_primary}
    for k in KS:
        topk = names[:k]
        hits = sum(1 for n in topk if n in relevant)
        out[f"precision@{k}"] = hits / k
        out[f"hit@{k}"] = gold_primary in topk
        out[f"top{k}"] = topk
    return out


def beta_binomial_posterior(wins, losses, n_samples=200000):
    """Posterior mean + 90% CI of p = P(win) under Beta(1+wins, 1+losses)."""
    import numpy as np
    rng = np.random.default_rng(0)
    a, b = 1 + wins, 1 + losses
    samples = rng.beta(a, b, size=n_samples)
    return {
        "posterior_mean": float(samples.mean()),
        "ci_5_95": [float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))],
        "wins": wins, "losses": losses,
    }


def main():
    with open(PREREG_PATH) as f:
        prereg = json.load(f)

    idx = ContextIndex(DEFAULT_MEMDIR)
    print(f"Corpus: N={idx.N} retrievable memory files, LSA rank chosen r={idx.lsa_rank} "
          f"(70% energy target), vocab={idx.V}")

    dev_tasks = prereg["task_set"]["dev_tasks"]
    held_tasks = prereg["task_set"]["held_out_tasks"]

    results = {"dev": {}, "held_out": {}}
    for group_name, tasks in [("dev", dev_tasks), ("held_out", held_tasks)]:
        for task in tasks:
            tid = task["id"]
            results[group_name][tid] = {"text": task["text"], "gold_primary": task["gold_primary"]}
            for method in METHODS_ALL:
                results[group_name][tid][method] = eval_task(idx, task, method)

    # ---- headline pre-registered comparison: TFIDF-cosine vs BM25, hit@3 ----
    def tally(group):
        wins = losses = ties_both_hit = ties_both_miss = 0
        rows = []
        for tid, r in results[group].items():
            t_hit = r["tfidf"]["hit@3"]
            b_hit = r["bm25"]["hit@3"]
            if t_hit and not b_hit:
                wins += 1; outcome = "TFIDF_WINS"
            elif b_hit and not t_hit:
                losses += 1; outcome = "BM25_WINS"
            elif t_hit and b_hit:
                ties_both_hit += 1; outcome = "TIE_BOTH_HIT"
            else:
                ties_both_miss += 1; outcome = "TIE_BOTH_MISS"
            rows.append((tid, outcome, r["tfidf"]["rank_primary"], r["bm25"]["rank_primary"]))
        return dict(wins=wins, losses=losses, ties_both_hit=ties_both_hit,
                    ties_both_miss=ties_both_miss, rows=rows)

    held_tally = tally("held_out")
    dev_tally = tally("dev")

    results["headline_tally_held_out_BLIND_TEST"] = held_tally
    results["headline_tally_dev_DISCLOSED_NONBLIND"] = dev_tally
    results["voi_posterior_held_out"] = beta_binomial_posterior(held_tally["wins"], held_tally["losses"])
    results["voi_posterior_dev_nonblind_sanitycheck"] = beta_binomial_posterior(dev_tally["wins"], dev_tally["losses"])

    # ---- secondary comparison: TFIDF-cosine vs naive-grep (the TRUE current-practice floor) ----
    def tally_vs_grep(group, method):
        wins = losses = ties_both_hit = ties_both_miss = 0
        rows = []
        for tid, r in results[group].items():
            m_hit = r[method]["hit@3"]
            g_hit = r["grep"]["hit@3"]
            if m_hit and not g_hit:
                wins += 1; outcome = f"{method.upper()}_WINS"
            elif g_hit and not m_hit:
                losses += 1; outcome = "GREP_WINS"
            elif m_hit and g_hit:
                ties_both_hit += 1; outcome = "TIE_BOTH_HIT"
            else:
                ties_both_miss += 1; outcome = "TIE_BOTH_MISS"
            rows.append((tid, outcome, r[method]["rank_primary"], r["grep"]["rank_primary"]))
        return dict(wins=wins, losses=losses, ties_both_hit=ties_both_hit,
                    ties_both_miss=ties_both_miss, rows=rows)

    results["secondary_tfidf_vs_naive_grep"] = {
        "dev": tally_vs_grep("dev", "tfidf"), "held_out": tally_vs_grep("held_out", "tfidf")}
    results["secondary_bm25_vs_naive_grep"] = {
        "dev": tally_vs_grep("dev", "bm25"), "held_out": tally_vs_grep("held_out", "bm25")}

    # ---- POST-HOC follow-up: does RRF-fusing BM25+TFIDF hedge away the h1-type
    # length-normalization failure without giving up the dev-set cosine wins?
    # (disclosed as post-hoc: motivated by, and only tested after, the h1 diagnosis) ----
    def tally_generic(group, method_a, method_b):
        wins = losses = ties_both_hit = ties_both_miss = 0
        rows = []
        for tid, r in results[group].items():
            a_hit, b_hit = r[method_a]["hit@3"], r[method_b]["hit@3"]
            if a_hit and not b_hit:
                wins += 1; outcome = f"{method_a.upper()}_WINS"
            elif b_hit and not a_hit:
                losses += 1; outcome = f"{method_b.upper()}_WINS"
            elif a_hit and b_hit:
                ties_both_hit += 1; outcome = "TIE_BOTH_HIT"
            else:
                ties_both_miss += 1; outcome = "TIE_BOTH_MISS"
            rows.append((tid, outcome, r[method_a]["rank_primary"], r[method_b]["rank_primary"]))
        return dict(wins=wins, losses=losses, ties_both_hit=ties_both_hit,
                    ties_both_miss=ties_both_miss, rows=rows)

    results["posthoc_fused_vs_bm25"] = {
        "dev": tally_generic("dev", "fused", "bm25"), "held_out": tally_generic("held_out", "fused", "bm25")}
    results["posthoc_fused_vs_tfidf"] = {
        "dev": tally_generic("dev", "fused", "tfidf"), "held_out": tally_generic("held_out", "fused", "tfidf")}
    results["posthoc_fused_vs_grep"] = {
        "dev": tally_generic("dev", "fused", "grep"), "held_out": tally_generic("held_out", "fused", "grep")}

    # ---- pooled (dev + held-out) hit@3 counts, n=9, for the honest combined picture ----
    pooled_hits = {m: sum(results[g][tid][m]["hit@3"] for g in ["dev", "held_out"] for tid in results[g])
                   for m in METHODS_ALL}
    results["pooled_hit@3_counts_n9"] = pooled_hits

    def pooled_tally(method_a, method_b):
        w = l = 0
        for g in ["dev", "held_out"]:
            for tid, r in results[g].items():
                a, b = r[method_a]["hit@3"], r[method_b]["hit@3"]
                w += int(a and not b); l += int(b and not a)
        return w, l

    w, l = pooled_tally("fused", "grep")
    results["voi_posterior_pooled_fused_vs_grep_n9"] = beta_binomial_posterior(w, l)
    w, l = pooled_tally("fused", "bm25")
    results["voi_posterior_pooled_fused_vs_bm25_n9"] = beta_binomial_posterior(w, l)
    w, l = pooled_tally("tfidf", "grep")
    results["voi_posterior_pooled_tfidf_vs_grep_n9"] = beta_binomial_posterior(w, l)

    # ---- OODA diagnostic: WHY did h1 flip against the pre-registered mechanism? ----
    # (document-length pathology: cosine fully divides out ||d||, so a long,
    # thoroughly-relevant doc can be OUT-SCORED by a short doc with partial overlap;
    # BM25's partial (b=0.75) length-normalization does not have this failure mode.)
    h1 = idx
    gold_h1 = "kinematic-singularity-is-sigma-min-jacobian-velocity-criticality-certifiable-on-real-robot-kin-spec.md"
    gi = idx.basenames.index(gold_h1)
    tf_scores = idx.score_tfidf_cosine(held_tasks[0]["text"])
    import numpy as np
    top1_i = int(np.argmax(tf_scores))
    results["ooda_diagnostic_h1_length_pathology"] = {
        "gold_doc": gold_h1, "gold_doclen": float(idx.doclen[gi]), "avgdl": float(idx.avgdl),
        "gold_tfidf_norm": float(np.linalg.norm(idx.X[gi])), "gold_cosine_score": float(tf_scores[gi]),
        "tfidf_top1_instead": idx.basenames[top1_i], "tfidf_top1_doclen": float(idx.doclen[top1_i]),
        "tfidf_top1_cosine_score": float(tf_scores[top1_i]),
        "mechanism": ("gold doc is ~7.4x average length and genuinely shares ALL 9 query content-words, "
                      "but its huge TF-IDF norm dilutes cosine similarity; a much shorter (300-word) doc "
                      "sharing only 2-3 of the 9 terms gets a higher cosine score because dividing by its "
                      "small norm concentrates that partial overlap -- a length-normalization inversion "
                      "that plain cosine is structurally prone to on a corpus with 127x doc-length range "
                      "(131-16534 words), and that BM25's partial length term does not suffer from.")
    }

    # aggregate precision@3 means per method, per group (for the full picture)
    agg = {}
    for group in ["dev", "held_out"]:
        agg[group] = {}
        for method in METHODS_ALL:
            vals = [results[group][tid][method]["precision@3"] for tid in results[group]
                    if isinstance(results[group][tid], dict) and method in results[group][tid]]
            agg[group][method] = {
                "mean_precision@3": sum(vals) / len(vals) if vals else None,
                "hit@3_count": sum(1 for tid in results[group] for m in [method]
                                   if results[group][tid][m]["hit@3"]),
                "n_tasks": len(vals),
            }
    results["aggregate_precision_at_3"] = agg

    with open(EVIDENCE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console report ----
    print("\n=== DEV SET (disclosed non-blind, used during design) ===")
    for tid, r in results["dev"].items():
        print(f"  {tid}: '{r['text'][:60]}...' gold={r['gold_primary']}")
        for m in METHODS_ALL:
            print(f"      {m:6s} rank={str(r[m]['rank_primary']):>4s}  P@1={r[m]['precision@1']:.2f} "
                  f"P@3={r[m]['precision@3']:.2f} P@5={r[m]['precision@5']:.2f}")

    print("\n=== HELD-OUT SET (blind pre-registered test) ===")
    for tid, r in results["held_out"].items():
        print(f"  {tid}: '{r['text'][:60]}...' gold={r['gold_primary']}")
        for m in METHODS_ALL:
            print(f"      {m:6s} rank={str(r[m]['rank_primary']):>4s}  P@1={r[m]['precision@1']:.2f} "
                  f"P@3={r[m]['precision@3']:.2f} P@5={r[m]['precision@5']:.2f}")

    print("\n=== HEADLINE: TF-IDF-cosine vs BM25, hit@3 ===")
    print("  HELD-OUT (the real test):", held_tally)
    print("  DEV (non-blind sanity check):", dev_tally)
    print("  P(TFIDF beats BM25) posterior [held-out]:", results["voi_posterior_held_out"])
    print("  P(TFIDF beats BM25) posterior [dev, non-blind]:", results["voi_posterior_dev_nonblind_sanitycheck"])
    print("\nPooled hit@3 counts (n=9, dev+held-out):", pooled_hits)
    print("\n=== POST-HOC: does RRF-fusion (bm25+tfidf) hedge the length-normalization flip? ===")
    print("  fused vs bm25  held_out:", results["posthoc_fused_vs_bm25"]["held_out"], " dev:", results["posthoc_fused_vs_bm25"]["dev"])
    print("  fused vs tfidf held_out:", results["posthoc_fused_vs_tfidf"]["held_out"], " dev:", results["posthoc_fused_vs_tfidf"]["dev"])
    print("  fused vs grep  held_out:", results["posthoc_fused_vs_grep"]["held_out"], " dev:", results["posthoc_fused_vs_grep"]["dev"])
    print("\nAggregate mean precision@3:", json.dumps(agg, indent=2))
    print("\nPooled (n=9) posteriors:")
    print("  P(fused beats grep):", results["voi_posterior_pooled_fused_vs_grep_n9"])
    print("  P(fused beats bm25):", results["voi_posterior_pooled_fused_vs_bm25_n9"])
    print("  P(tfidf beats grep):", results["voi_posterior_pooled_tfidf_vs_grep_n9"])
    print(f"\nEvidence written to {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
