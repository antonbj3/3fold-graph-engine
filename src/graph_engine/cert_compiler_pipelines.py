#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cert_compiler_pipelines.py -- [E0712F] pipeline DESCRIPTIONS fed to cert_compiler.compile_pipeline.

Each description encodes ONLY structural facts about the pipeline (stage forward-model/cert-type,
which physical DOF each stage's argument carries, the ambient gauge the substrate cares about) --
NOT any hand-built fix (no G4/G5/composed-1/composed-2/observed-generated-split is declared here as
an existing facet; those are exactly what cert_compiler_score.py checks the COMPILER recommends, then
diffs against the real evidence). Every structural fact below is cited to the real source/PREREG file
it comes from, so this is a description of the SUBSTRATE, not a restatement of the answer.

4 pipelines: 3 REAL (hand-built + QC'd this window) + 1 NULL CONTROL (a non-degenerate synthetic
pipeline with NO genuine gap, used to force the adversary "the compiler recommends fixes unconditionally
regardless of structure" -- see cert_compiler_PREREG.json for why this control is mandatory).
"""
import os

# =====================================================================================================
# PIPELINE 1 -- dcompose_v2 / dcompose_law / dcompose_axisflip: vision(recon) -> sim-readiness, 3 real
# committed stages (facts cited to dcompose_v2_PREREG.json + dcompose_v2_pipeline.py +
# dcompose_axisflip_falsealarm.py). This description is the PRE-FIX state (before
# G4/G5 were hand-built) -- i.e. it contains ONLY stage1/stage2/stage3's own native cert facets.
# =====================================================================================================
PIPE_DCOMPOSE = {
    "pipeline_id": "dcompose_vision_to_sim",
    "argument_kind": "point_set_R3",
    "ambient_gauge": [
        # scope declared (domain fact, named up front): for a "is this vision-recon->sim assembly
        # physically correct" question, the gauge concerns that matter are (i) overall metric SCALE
        # (a units bug can silently rescale the whole assembly) and (ii) REFLECTION/chirality (a
        # mirror-flipped part is a real, distinct assembly defect) and (iii) the OCCLUSION domain-null
        # for this exterior-view reconstruction stage.
        {"tag": "geometric_scale", "kind": "shared_dof_continuous"},
        {"tag": "REFLECTION", "kind": "discrete_parity"},
        {"tag": "occlusion_interior", "kind": "domain_projection",
         "channels": ["stage2_recon_cert_vector"]},  # ONE exterior-view reconstruction channel only
    ],
    "stages": [
        {
            # pi_slice1_provenance_cert.py, reused verbatim as stage 1 (dcompose_v2_PREREG.json
            # composed_legs_committed_readonly.stage1_image_provenance). Certifies UPSTREAM VISUAL
            # evidence trustworthiness via a linear-projector null-space split (see PIPE_PISLICE1
            # below for ITS OWN structural derivation) -- carries NEITHER geometric_scale nor
            # chirality (it operates on a 2D pixel-grid null-space, a DIFFERENT modality/argument
            # entirely; dcompose_v2_PREREG.json's honest_scope_carried_from_upstream names stage1 as
            # "a parallel, decorrelated gate", not a producer of the geometry chain's scale/chirality
            # facts) -- deliberately given NO cert_facets carrying the geometry ambient_gauge tags.
            "name": "stage1_image_provenance",
            "cert_facets": [],
        },
        {
            # batch recon_cert_vector_v2 (dcompose_v2_PREREG.json stage2_recon_cert_vector): its
            # metric_anchor computes scale_factor = known_distance_true_m / ||P0n[ib]-P0n[ia]|| --
            # a ratio of two Euclidean distances (one known/real, one measured in the recon-space
            # point cloud) -- textbook distance_ratio. It ALSO computes an occlusion_bound from the
            # SAME single (synthetic-motion, but geometrically single-view-style) reconstruction.
            "name": "stage2_recon_cert_vector",
            "cert_facets": [
                {"cert_type": "distance_ratio", "carries": ["geometric_scale"]},
                {"cert_type": "reconstruction_fidelity", "carries": ["occlusion_interior"]},
            ],
        },
        {
            # batch cad_sim_ready (dcompose_v2_PREREG.json stage3_sim_readiness): computes its OWN
            # units-gate scale = known_length/measured (a 2nd, independent distance_ratio -- built
            # from the SAME real known_length fact but a DIFFERENT measurement, the post-glue mesh)
            # PLUS a kappa=sigma_max/sigma_min conditioning ratio (quadratic_spectrum_ratio) and a raw
            # sigma_min mechanism check (quadratic_spectrum_absolute). Both spectrum facets carry
            # geometric_scale+chirality as ambient tags their ARGUMENT (mesh vertices) varies over,
            # per their ontology invariance class (see cert_compiler.ONTOLOGY citations).
            "name": "stage3_sim_readiness",
            "cert_facets": [
                {"cert_type": "distance_ratio", "carries": ["geometric_scale"]},
                {"cert_type": "quadratic_spectrum_ratio", "carries": ["geometric_scale", "REFLECTION"]},
                {"cert_type": "quadratic_spectrum_absolute", "carries": ["geometric_scale", "REFLECTION"]},
            ],
        },
    ],
    # supplementary static-analysis targets (NOT part of the group-theoretic derivation above -- this
    # is the SEPARATE engineering-level question "does stage3 THRESHOLD its own units-gate ratio, or
    # only compute+report it?", answered by grepping the REAL source file, exactly as dcompose_v2's
    # own forced-adversary section did by direct code read).
    "static_checks": [
        {"claim": "stage3_units_gate_computed_not_gated",
         "source_path": os.environ.get("PIPELINE_SOURCE_ROOT", ".") + "/"
                         "wave501_cad_sim_ready_ordered_pregate_topology_units_before_numeric_sigmin_kappa_composable_tool.py",
         "quantity_token": "scale"},
    ],
}


# =====================================================================================================
# PIPELINE 2 -- optics_compose: kk_cert (real (n,k) dispersion) -> ir_material (MAP/Fisher BRDF fit)
# -> ir_chroma (chroma reconstruction). Facts cited to optics_compose_PREREG.json / optics_compose_lib.py
# / kk_cert_lib.py / ir_material_lib.py / ir_chroma_lib.py.
# =====================================================================================================
PIPE_OPTICS = {
    "pipeline_id": "optics_compose_kk_material_chroma",
    "argument_kind": None,  # not a point-set/group-action substrate -- a shared SCALAR physical DOF (F0)
    "ambient_gauge": [
        # the physical DOF every stage below touches in some form: the normal-incidence Fresnel
        # reflectance F0 of the real material. Declared (domain fact: F0(n,k) bridges stage A's
        # dispersion data to stage B/C's BRDF parameter -- the BRIDGE EQUATION itself is optics,
        # supplied, NOT derived by this compiler; see PREREG honest-boundary).
        {"tag": "F0_reflectance", "kind": "shared_dof_continuous"},
    ],
    "stages": [
        {
            # kk_cert_lib.py: Kramers-Kronig causality self-consistency (interior_mask / subtractive_
            # predict / relative_residual) on REAL (n,k) dispersion data. This is the stage that
            # actually PRODUCES an independent, externally-measured estimate of F0 (via the Fresnel
            # bridge fresnel_F0(n,k), computed in optics_compose_lib.py from kk_cert's own loaded real
            # (n,k)) -- but the causality gate ITSELF is a functional of (n,k) alone, decorrelated from
            # F0/Fresnel by construction (verified: kk_cert_lib.py never references fresnel/F0/reflect
            # -- see static_checks below).
            "name": "stageA_kk_causality",
            "cert_facets": [
                {"cert_type": "causality_transform", "carries": []},        # its OWN native gate: blind to F0 by construction
                {"cert_type": "distance_ratio", "carries": ["F0_reflectance"]},  # the Fresnel-bridge F0(n,k) IS a real, independent estimate of the DOF (treated as a ratio-like real-valued estimate for compiler purposes: it is a deterministic function of REAL measured (n,k), i.e. an externally-anchored channel)
            ],
        },
        {
            # ir_material_lib.py map_fit_5_batch + Fisher/posterior (optics_compose_lib.stage_B_fit):
            # fits F0_map from a SYNTHETIC angle-sweep rendered FROM the SAME F0_gt it then checks
            # itself against (G_self_consistent_pass = |F0_map-F0_gt|<=3sigma) -- a self-consistency
            # loop by construction.
            "name": "stageB_ir_material_fit",
            "cert_facets": [
                {"cert_type": "self_consistency_map", "carries": ["F0_reflectance"]},
            ],
        },
        {
            # ir_chroma_lib.py guided-filter chroma reconstruction (optics_compose_lib.stage_C_cell):
            # RRR_edge / reconstruction-floor Delta_E76 are round-trip fidelity certs (render -> 4:2:0
            # subsample -> guided reconstruct -> compare to the SAME render) -- self-referential.
            "name": "stageC_ir_chroma_render",
            "cert_facets": [
                {"cert_type": "reconstruction_fidelity", "carries": ["F0_reflectance"]},
            ],
        },
    ],
    "static_checks": [
        {"claim": "stageA_decorrelated_from_F0_by_nonreference",
         "source_path": "examples/optical_constants/kk_cert_lib.py",
         "target_tokens": ["fresnel", "f0", "reflect"]},
        {"claim": "disjoint_import_graphs",
         "source_paths": ["examples/optical_constants/kk_cert_lib.py",
                           "ir_material_lib.py",
                           "ir_chroma_lib.py"],
         "local_lib_names": ["kk_cert_lib", "ir_material_lib", "ir_chroma_lib",
                             "kramers_kronig_passivity_cert", "ir_matcap1_brdf_identifiability"]},
    ],
}


# =====================================================================================================
# PIPELINE 3 -- pi_slice1 (image provenance): a SINGLE linear forward-model stage, not a multi-stage
# gauge-group composition -- tests the compiler's OTHER derivation branch (rank-nullity), not the
# ambient-gauge branch. Facts cited to pi_slice1_provenance_cert.py / pi_slice1_PREREG.json.
# =====================================================================================================
PIPE_PISLICE1 = {
    "pipeline_id": "pi_slice1_image_provenance",
    "argument_kind": None,
    "ambient_gauge": [],
    "stages": [{"name": "pi_slice1_D_cert", "cert_facets": []}],
    "linear_forward_models": [
        {
            "stage": "pi_slice1_D_cert",
            # D = build_D(H,W,s,sigma): an explicit sparse LINEAR blur+decimate operator (declared
            # structural fact: it is literally a matrix, sparse-csr, applied by cert.D @ v) --
            "type": "linear_measurement",
            "family_param": ["s", "sigma"],
            # the SAME operator family reduces to (near-)identity at s=1 (declared: pi_slice1's own
            # s1_control docstring: "s=1, tiny sigma (D ~ identity)").
            "identity_at": {"s": 1},
            # the pipeline explicitly runs BOTH a fixed-sigma main cert AND a PSF-band sweep
            # perturbing the ASSUMED sigma away from the TRUE one (declared: DESIGN["psf_band_
            # perturbations_px"] exists) -- i.e. model mismatch is a named, tested possibility.
            "model_mismatch_possible": True,
        }
    ],
}


# =====================================================================================================
# NULL CONTROL -- a synthetic, non-degenerate pipeline with GENUINELY NO gap: 2 stages independently
# measuring UNRELATED DOF (no shared_dof_continuous with <2 cross-checks), one discrete_parity
# generator that IS already broken by an existing signed_volume facet, one domain_projection tag
# ALREADY covered by 2 independent channels. Forces the adversary "the compiler recommends fixes
# unconditionally regardless of structure" -- if it fires here, the compiler is a rubber stamp.
# =====================================================================================================
PIPE_NULL_CONTROL = {
    "pipeline_id": "NULL_CONTROL_no_genuine_gap",
    "argument_kind": "point_set_R3",
    "ambient_gauge": [
        {"tag": "dof_X", "kind": "shared_dof_continuous"},          # only 1 producer below -> WOULD be
                                                                      # joint-blind (needs a 2nd channel,
                                                                      # honestly still flagged -- this is
                                                                      # the "1 producer" sub-branch, kept
                                                                      # to show the control ISN'T rigged
                                                                      # to always emit zero recommendations
        {"tag": "dof_Y", "kind": "shared_dof_continuous"},           # 2 producers, ALREADY cross-checked
        {"tag": "parity_Z", "kind": "discrete_parity"},              # ALREADY broken by a signed_volume facet
        {"tag": "interior_W", "kind": "domain_projection",
         "channels": ["stageP", "stageQ"]},                          # 2 independent channels already
    ],
    "stages": [
        {"name": "stageP", "cert_facets": [
            {"cert_type": "distance_ratio", "carries": ["dof_X"]},
            {"cert_type": "distance_ratio", "carries": ["dof_Y"]},
            {"cert_type": "signed_volume", "carries": ["parity_Z"]},
        ]},
        {"name": "stageQ", "cert_facets": [
            {"cert_type": "distance_ratio", "carries": ["dof_Y"]},
            {"cert_type": "cross_stage_consistency", "carries": ["dof_Y"]},
        ]},
    ],
}


ALL_PIPELINES = {
    "dcompose": PIPE_DCOMPOSE,
    "optics": PIPE_OPTICS,
    "pislice1": PIPE_PISLICE1,
    "null_control": PIPE_NULL_CONTROL,
}
