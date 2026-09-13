#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cert_compiler.py -- [E0712F] THE CERT COMPILER, v0.

Turns a PIPELINE DESCRIPTION (stages + their cert facets + the ambient gauge group the
substrate's correctness depends on) into a COMPILED cert-vector spec: predicted per-stage
blind axes, predicted COMPOSED checks (with their required invariant CLASS, not just "add a
check"), and the predicted blind-set codimension before/after. This is NOT a black box that
recommends "add more certs" unconditionally -- every recommendation is a deduction from a
declared structural fact (an invariance-group table entry, an unbroken generator, or a static
grep of real source), so it can and must ALSO emit "no composed check needed" on a
non-degenerate NULL pipeline (see cert_compiler_pipelines.py's PIPE_NULL_CONTROL).

THE THEOREM BASE (compress-not-accrete -- this compiler does not invent new physics, it
implements three reference laws, cited inline):
  (1) blind-spot-two-kinds law (C body-888/889/891): a cert's blindness is either
      GROUP-ACTION (an unbroken generator of the ambient gauge -> anchor OUTSIDE the group)
      or DOMAIN/PROJECTION (upstream map non-injective / self-referential -> a SEPARATE
      measurement channel). This compiler's per-generator rule (2) below IS that law,
      mechanized: for each ambient generator, ask "does ANY stage's cert break it?" and if
      not, classify by generator KIND and emit the matching anchor TYPE.
  (2) invariant theory / rank-nullity (textbook, not domain insight): a ratio/distance-based
      cert is invariant to the full orthogonal group O(n) (rotation+reflection) by construction
      (Euclidean length is O(n)-invariant); the unique (up to scale) low-degree invariant that
      is ODD under an improper (det=-1) generator is the alternating multilinear form -- the
      determinant / signed volume in R^3. A LINEAR forward operator D splits R^N = row(D) (+)
      null(D) EXACTLY (fundamental theorem of linear algebra) -- so a linear-measurement stage
      structurally requires EXACTLY a 2-axis {observed, generated} cert vector, no more no less,
      derivable the moment "D is linear" is declared, before any data is seen.
  (3) over-determination / n_eff (neff-canon, body-891): if >=2 stages each independently
      carry/estimate the SAME physical DOF via DIFFERENT data or functionals, a cross-stage
      residual comparing them is a NEW invariant that exists only in the composition (this is
      the dcompose-law's G4 / optics_compose's COMPOSED-1 pattern, generalized).

HONEST SCOPE (declared before any scoring number is computed -- see cert_compiler_PREREG.json):
this compiler predicts the CLASS of fix needed (cross-stage-consistency vs signed-invariant vs
external-anchor) and its target generator/DOF -- it does NOT predict calibration constants
(thresholds, exposure gains, guided-filter r/eps, injection-scaling magnitudes) or the
domain-specific BRIDGE EQUATION between two stages' native parametrizations (e.g. Fresnel
F0(n,k) -- that a photon's F0 and Fresnel's (n,k) describe the "same physical DOF" at all is
optics, supplied as a declared fact, not derived from bare structure). Those are pre-registered
as OUT OF COMPILER SCOPE, not silently missed.
"""
import ast
import os

# =====================================================================================================
# ONTOLOGY -- cert TYPE -> its structural invariance class. Substrate-independent: these are facts
# about the MATH of the cert construction (a ratio-of-distances, a quadratic-form spectrum, a linear
# projector, a self-consistency loop, a reconstruction-fidelity loop), not about any one pipeline.
# =====================================================================================================
ONTOLOGY = {
    "distance_ratio": {
        "invariant_to": {"ROTATION", "REFLECTION", "TRANSLATION"},
        "cite": "Euclidean length ||x|| is invariant under the full orthogonal group O(n) + translation "
                "-- a ratio of two lengths is invariant to everything O(n)+translation is, and SENSITIVE "
                "only to overall SCALE (that is its designed job).",
    },
    "quadratic_spectrum_ratio": {
        "invariant_to": {"SCALE", "ROTATION", "REFLECTION", "TRANSLATION"},
        "cite": "kappa = sigma_max/sigma_min of a symmetric edge-vector-built matrix K: K(alpha*R*x) = "
                "alpha^2 R K R^T for any orthogonal R (incl. reflection) and scalar alpha -- kappa is a "
                "RATIO of eigenvalues of a matrix that scales uniformly and conjugates orthogonally, so "
                "kappa is invariant to SCALE, ROTATION, REFLECTION, TRANSLATION: it sees SHAPE only.",
    },
    "quadratic_spectrum_absolute": {
        "invariant_to": {"ROTATION", "REFLECTION", "TRANSLATION"},
        "cite": "sigma_min itself (not a ratio) scales linearly with overall SCALE (sigma_min(alpha*K) = "
                "alpha^2*sigma_min(K) for the stiffness-K convention here) -- invariant to ROTATION/"
                "REFLECTION/TRANSLATION (orthogonal conjugation preserves the spectrum) but NOT to SCALE.",
    },
    "signed_volume": {
        "invariant_to": {"ROTATION", "TRANSLATION"},
        "cite": "the triple product / signed tetrahedron volume is the degree-3 ALTERNATING multilinear "
                "form on 4 affinely-independent points: invariant under SO(3) (proper rotation, det=+1) + "
                "translation, but changes SIGN under any single improper (det=-1, e.g. a coordinate "
                "reflection) generator, and scales as alpha^3 under uniform scale alpha -- the unique "
                "(up to a scalar power) low-degree invariant that BREAKS reflection while a distance-based "
                "cert cannot (distance is even under reflection: ||Rx-Ry||=||x-y|| for R orthogonal of "
                "EITHER determinant sign).",
    },
    "linear_projector_nullspace": {
        "invariant_to": {"NULLSPACE_D"},
        "cite": "fundamental theorem of linear algebra: R^N = row(D) (+)_orthogonal null(D) EXACTLY for "
                "any real D -- a cert built as the exact orthogonal projector P onto row(D) (P=D^T(DD^T)^-1 D) "
                "detects everything in row(D) and is exactly, structurally blind to null(D); no 3rd axis "
                "is needed or possible for a single linear D (rank-nullity is exhaustive).",
    },
    "self_consistency_map": {
        "invariant_to": {"EXTERNAL_TRUTH_OF_PRIOR"},
        "cite": "a MAP-fit self-consistency check (does my estimator recover the SAME ground truth I used "
                "to generate its own synthetic training/test render?) is, by the structure of the loop "
                "(generate from GT -> fit -> compare to the SAME GT), blind to whether that GT itself "
                "equals an INDEPENDENT external truth -- it certifies internal recoverability/"
                "identifiability, not external correctness (verdict-blind-set > value-blind-set).",
    },
    "reconstruction_fidelity": {
        "invariant_to": {"EXTERNAL_TRUTH_OF_INPUT"},
        "cite": "a round-trip fidelity cert (||recon(measured) - measured||, or a ratio thereof vs a "
                "weaker baseline) is defined RELATIVE TO ITS OWN INPUT -- it is structurally blind to "
                "whether that input is itself physically correct, since the same reconstruction quality "
                "obtains whether the input was rendered from a true or a wrong physical parameter.",
    },
    "causality_transform": {
        "invariant_to": {"UNREFERENCED_FUNCTIONAL"},
        "cite": "a self-consistency cert over an internal integral-transform relation (e.g. Kramers-Kronig "
                "n<->k) is a functional of ONLY the variables its formula references; STRUCTURALLY blind "
                "to any OTHER functional of those same variables it never reads (verified per-pipeline by "
                "static source grep here, not assumed) -- decorrelated-by-construction, not by luck.",
    },
    "cross_stage_consistency": {
        "invariant_to": set(),
        "cite": "a cross-stage residual comparing two INDEPENDENT stages' estimates of the same declared "
                "physical DOF breaks (by definition) exactly the DOF it targets -- this is the type the "
                "compiler ITSELF recommends when it finds >=2 producers of one DOF (rule 3 below); it does "
                "not pre-exist as a per-stage cert.",
    },
}

# lowest-degree ODD (alternating) invariant, keyed by (argument_kind, generator_kind) -- invariant
# theory, substrate-independent (would extend to R^2 signed-area, R^4 4-form, etc.)
ALTERNATING_INVARIANT_TABLE = {
    ("point_set_R3", "discrete_parity"): dict(
        recommended_cert_type="signed_volume",
        construction="triple product / signed tetrahedron volume of 4 affinely-independent points "
                     "(degree-3 alternating form, odd under one reflection, even under SO(3))."),
    ("point_set_R2", "discrete_parity"): dict(
        recommended_cert_type="signed_area",
        construction="2D cross product (signed area) of 3 affinely-independent points (degree-2 "
                     "alternating form, odd under one reflection, even under SO(2))."),
}


# =====================================================================================================
# STATIC-ANALYSIS HELPERS -- mechanical facts about real source files, NOT eyeballed. Both checks
# below reproduce (generalized, reusable) the exact ad-hoc greps the hand-built pipelines already ran
# themselves (dcompose_v2's "confirmed by direct code read" + optics_compose's kk_cert_lib token grep)
# -- i.e. this IS a compilable component, not a hand-wave.
# =====================================================================================================
def static_computed_not_gated(source_path, quantity_token):
    """Is `quantity_token` ever compared against a literal/threshold anywhere in source_path (a
    real gate), or only assigned+reported (a computed-but-unused number)? Heuristic (token +
    comparison-operator co-occurrence after assignment) -- same technique dcompose_v2_PREREG.json's
    forced-adversary section used by direct code read; here mechanized as a reusable grep."""
    if not os.path.isfile(source_path):
        return {"error": f"source not found: {source_path}", "gated": None}
    src = open(source_path).read()
    assigned = f"{quantity_token}=" in src.replace(" ", "")
    comparison_tokens = [f"{quantity_token}<", f"{quantity_token}>", f"{quantity_token}==",
                         f"abs({quantity_token}", f"if{quantity_token}", f"{quantity_token}!="]
    compact = src.replace(" ", "").replace("\n", "")
    gated = any(tok in compact for tok in comparison_tokens)
    return {"assigned": assigned, "gated": bool(gated), "source_path": source_path,
            "quantity_token": quantity_token}


def static_decorrelated_by_nonreference(source_path, target_tokens):
    """Does source_path reference ANY of target_tokens at all (case-insensitive)? Used to verify a
    'decorrelated by construction' claim (e.g. kk_cert_lib.py never mentions Fresnel/F0/reflectance)
    mechanically rather than asserting it."""
    if not os.path.isfile(source_path):
        return {"error": f"source not found: {source_path}", "decorrelated": None}
    src = open(source_path).read().lower()
    hits = [t for t in target_tokens if t.lower() in src]
    return {"decorrelated": len(hits) == 0, "hits": hits, "source_path": source_path}


def static_disjoint_imports(paths, local_lib_names):
    """AST-parse each file's imports; True iff no pair shares a LOCAL lib import (no relabeled-
    anchor / shared-upstream confound at the import-graph level). Generalizes optics_compose's own
    disjoint-import-graph forced-adversary check."""
    import_sets = []
    for p in paths:
        tree = ast.parse(open(p).read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        import_sets.append(mods & set(local_lib_names))
    all_disjoint = all((import_sets[i] & import_sets[j] == set())
                        for i in range(len(import_sets)) for j in range(i + 1, len(import_sets)))
    return {"per_file_local_imports": dict(zip(paths, [sorted(s) for s in import_sets])),
            "all_pairwise_disjoint": bool(all_disjoint)}


# =====================================================================================================
# THE COMPILER
# =====================================================================================================
def compile_pipeline(desc):
    """desc: dict with keys
         pipeline_id, argument_kind (for the discrete_parity branch; None if not applicable),
         ambient_gauge: [ {tag, kind,...} ]  three DISTINCT kinds, three DISTINCT joint-blindness
             criteria (deliberately NOT collapsed into one rule -- they are different laws):
           - "discrete_parity": a Z2 generator (e.g. REFLECTION). Joint-blind iff every facet
             that CARRIES this tag lists it in invariant_to (rule A: unbroken generator, blind-spot-
             two-kinds GROUP-ACTION case). Needs {tag} in each carrying facet's "carries" list.
           - "shared_dof_continuous": a physical DOF >=2 stages independently estimate via different
             data (e.g. geometric scale). Joint-blind iff (>=2 stages carry a facet targeting this
             tag) AND (no EXISTING facet of cert_type "cross_stage_consistency" already targets it) --
             rule B: this is a COMPOSITION-CONSISTENCY gap, not a group-action orbit, so it does NOT
             matter whether each stage's OWN facet "breaks" the tag against ITS OWN local reference;
             what matters is whether the two producers are ever compared to EACH OTHER.
           - "domain_projection": a non-injective upstream map / self-referential cert (e.g.
             exterior-view occlusion). Joint-blind iff fewer than 2 INDEPENDENT declared channels
             ("channels": [stage names]) cover this tag -- rule C: fixed only by a 2nd independent
             measurement channel, never by a cleverer functional of the SAME data (blind-spot-two-
             kinds DOMAIN case).
         stages: [ {name, cert_facets: [ {cert_type, carries: [tags]} ]} ]
         linear_forward_models: [ {stage, family_param: [...], identity_at: {...},
             model_mismatch_possible: bool} ]  (optional -- triggers the independent rank-nullity
             derivation, orthogonal to the ambient-gauge branch above)
    Returns a compiled report: predicted joint-blind generators, recommendations (typed, not generic),
    predicted codim before/after, + linear-forward-model axis predictions if declared.
    """
    def build_facets(stages_):
        out_facets = []
        for st in stages_:
            for f in st.get("cert_facets", []):
                ctype = f["cert_type"]
                inv = set(ONTOLOGY[ctype]["invariant_to"])
                out_facets.append({"stage": st["name"], "cert_type": ctype, "carries": set(f.get("carries", [])),
                                    "invariant_to": inv})
        return out_facets

    def analyze(facets, ambient_, argument_kind):
        """The 3 rules (A/B/C), factored out so codim_after can be MEASURED by re-running this
        same analysis on an 'adopted' facet/channel list, not assumed to be 0."""
        joint_blind_, recs_ = [], []
        for g in ambient_:
            tag, kind = g["tag"], g["kind"]

            if kind == "discrete_parity":
                relevant = [f for f in facets if tag in f["carries"]]
                breaks = [f for f in relevant if tag not in f["invariant_to"]]
                if relevant and not breaks:
                    joint_blind_.append(g)
                    key = (argument_kind, "discrete_parity")
                    entry = ALTERNATING_INVARIANT_TABLE.get(key)
                    if entry:
                        recs_.append({"tag": tag, "kind": kind,
                                      "reason": f"generator '{tag}' is discrete-parity and NO existing "
                                                f"facet (of {sorted(set(f['stage'] for f in relevant))}) "
                                                f"breaks it (all listed as invariant_to it)",
                                      "recommended_cert_type": entry["recommended_cert_type"],
                                      "construction": entry["construction"]})
                    else:
                        recs_.append({"tag": tag, "kind": kind,
                                      "reason": "discrete-parity generator unbroken, but no invariant-"
                                                "theory table entry for this argument_kind",
                                      "recommended_cert_type": "UNKNOWN"})

            elif kind == "shared_dof_continuous":
                producers = sorted(set(f["stage"] for f in facets if tag in f["carries"]))
                already_cross_checked = any(f["cert_type"] == "cross_stage_consistency" and tag in f["carries"]
                                             for f in facets)
                if len(producers) >= 2 and not already_cross_checked:
                    joint_blind_.append(g)
                    recs_.append({"tag": tag, "kind": kind,
                                  "reason": f"{len(producers)} stages independently carry/estimate "
                                            f"this DOF via different data/functionals: {producers} "
                                            f"-- each may be internally self-consistent yet the PAIR "
                                            f"never compared (a glue-layer defect between them is "
                                            f"invisible to either stage's OWN gate)",
                                  "recommended_cert_type": "cross_stage_consistency",
                                  "stages": producers})
                elif len(producers) == 1:
                    joint_blind_.append(g)
                    recs_.append({"tag": tag, "kind": kind,
                                  "reason": "only 1 stage carries this DOF -- a 2nd INDEPENDENT "
                                            "channel is required, not derivable further from "
                                            "structure alone",
                                  "recommended_cert_type": "EXTERNAL_ANCHOR_NEEDED"})
                # else: >=2 producers AND already cross-checked -> NOT joint-blind, no recommendation
                # (this is the branch a non-degenerate NULL pipeline must exercise -- PIPE_NULL_CONTROL)

            elif kind == "domain_projection":
                channels = g.get("channels", [])
                if len(channels) < 2:
                    joint_blind_.append(g)
                    recs_.append({"tag": tag, "kind": kind,
                                  "reason": f"only {len(channels)} independent channel(s) ({channels}) "
                                            "cover this tag -- per blind-spot-two-kinds law, a "
                                            "domain/projection null needs a SEPARATE measurement "
                                            "channel, not an in-group anchor or a cleverer functional "
                                            "of the SAME data",
                                  "recommended_cert_type": "EXTERNAL_ANCHOR_NEEDED (separate channel)"})
        return joint_blind_, recs_

    stages = desc.get("stages", [])
    ambient = desc.get("ambient_gauge", [])
    argument_kind = desc.get("argument_kind")
    facets = build_facets(stages)
    joint_blind, recommendations = analyze(facets, ambient, argument_kind)
    codim_before = len(joint_blind)

    # codim_after: MEASURED, not assumed -- mechanically construct the 'adopted' facets/ambient-gauge
    # (append exactly the facet/channel each recommendation above says to add) and RE-RUN the identical
    # analyze on it. If the 3 rules above are self-consistent, this must independently come out at 0;
    # if it didn't, that would be a real bug in the rules, not a tautology (nothing forces this to be 0
    # other than re-running the same 3-rule analysis a 2nd time on the augmented facts).
    adopted_facets = list(facets)
    adopted_ambient = []
    for g in ambient:
        g2 = dict(g)
        rec = next((r for r in recommendations if r["tag"] == g["tag"]), None)
        if rec is None:
            adopted_ambient.append(g2)
            continue
        if rec["recommended_cert_type"] in ("signed_volume", "signed_area"):
            adopted_facets.append({"stage": "__adopted_fix__", "cert_type": rec["recommended_cert_type"],
                                    "carries": {g["tag"]}, "invariant_to": set(ONTOLOGY.get(
                                        rec["recommended_cert_type"], {"invariant_to": set()})["invariant_to"])})
        elif rec["recommended_cert_type"] == "cross_stage_consistency":
            adopted_facets.append({"stage": "__adopted_fix__", "cert_type": "cross_stage_consistency",
                                    "carries": {g["tag"]}, "invariant_to": set()})
        elif rec["recommended_cert_type"].startswith("EXTERNAL_ANCHOR_NEEDED"):
            if g["kind"] == "domain_projection":
                g2["channels"] = list(g.get("channels", [])) + ["__adopted_2nd_channel__"]
            elif g["kind"] == "shared_dof_continuous":
                adopted_facets.append({"stage": "__adopted_2nd_producer__", "cert_type": "distance_ratio",
                                        "carries": {g["tag"]}, "invariant_to": set(ONTOLOGY["distance_ratio"]["invariant_to"])})
                adopted_facets.append({"stage": "__adopted_fix__", "cert_type": "cross_stage_consistency",
                                        "carries": {g["tag"]}, "invariant_to": set()})
        adopted_ambient.append(g2)
    joint_blind_after, _ = analyze(adopted_facets, adopted_ambient, argument_kind)
    codim_after_measured = len(joint_blind_after)

    out = {"pipeline_id": desc["pipeline_id"], "n_facets_declared": len(facets),
           "joint_blind_generators": joint_blind, "recommendations": recommendations,
           "codim_before": codim_before, "codim_after_if_recommendations_adopted": codim_after_measured}

    lfms = desc.get("linear_forward_models", [])
    if lfms:
        lfm_out = []
        for lfm in lfms:
            axes = ["observed (component in row(D_assumed^T), the range of the certified forward "
                    "operator -- exact, rank-nullity)",
                    "generated (component in null(D_assumed), the orthogonal complement -- exact, "
                    "rank-nullity)"]
            preds = {"stage": lfm["stage"], "predicted_axes": axes,
                     "predicted_n_axes": 2,
                     "predicted_experiments": []}
            if lfm.get("model_mismatch_possible"):
                preds["predicted_experiments"].append(
                    "model-mismatch sweep: perturb the ASSUMED operator's family_param away from the "
                    "TRUE one and re-measure the missed/leaked fraction -- range(D_true^T) and "
                    "range(D_assumed^T) are DIFFERENT subspaces under mismatch (principal-angle / "
                    "subspace-perturbation argument), so content in null(D_true) can leak into "
                    "row(D_assumed); predicted MONOTONIC growth with |perturbation| (assuming D varies "
                    "continuously in its family_param, a smoothness assumption declared, not derived).")
            if lfm.get("identity_at"):
                preds["predicted_experiments"].append(
                    f"known-answer control at family_param={lfm['identity_at']}: the SAME operator "
                    f"FAMILY reduces to (near-)identity there (declared), so null(D) -> {{0}} and the "
                    f"cert's own false-positive rate on a TRIVIAL instance is directly measurable.")
            lfm_out.append(preds)
        out["linear_forward_model_predictions"] = lfm_out

    return out


def print_report(report):
    print(f"\n{'='*100}\nCOMPILED CERT-VECTOR SPEC -- {report['pipeline_id']}\n{'='*100}")
    print(f"  facets declared: {report['n_facets_declared']}")
    print(f"  joint-blind generators (codim_before={report['codim_before']}):")
    for g in report["joint_blind_generators"]:
        print(f"    - {g['tag']} (kind={g['kind']})")
    print(f"  recommendations ({len(report['recommendations'])}):")
    for r in report["recommendations"]:
        print(f"    - [{r['tag']}] -> {r['recommended_cert_type']}  :: {r['reason']}")
    if "linear_forward_model_predictions" in report:
        for lfm in report["linear_forward_model_predictions"]:
            print(f"  linear-forward-model stage '{lfm['stage']}': predicted_n_axes={lfm['predicted_n_axes']}")
            for a in lfm["predicted_axes"]:
                print(f"    axis: {a}")
            for e in lfm["predicted_experiments"]:
                print(f"    experiment: {e}")
    print(f"  codim_after_if_recommendations_adopted={report['codim_after_if_recommendations_adopted']}")
