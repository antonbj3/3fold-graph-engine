#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cert_compiler_prospective_pazy_pipeline.py -- compiler_prospective, PIPELINE #4 (PROSPECTIVE,
BLIND at authoring time).

WHY THIS FILE EXISTS: cert_compiler_v0's 6/6 recovery (cert_compiler_PREREG.json) was against 3 pipelines
whose ontology (cert_compiler.py) was written AFTER reading them (disclosed caveat) -- that is a
RECONSTRUCTION-ADEQUACY test, not a predictive one. This file is the PROSPECTIVE test: a 4th REAL,
hand-built, committed-evidence pipeline the compiler's ontology has never seen, drawn from an unrelated
lane (G, aeroelastic flutter) and an unrelated substrate (frequency-domain FRF damping/coalescence, not
vision-CAD geometry or optics BRDF). cert_compiler.py itself is NOT edited (same frozen file D already
uses for pislice1/dcompose/optics).

SUBSTRATE: real Delft Pazy Wing wind-tunnel FRF archive (de Boer/Karpel/Sodja, AIAA 2023-0379, 4TU repo),
loaded directly from the public Delft Pazy flutter benchmark .mat files ($PAZY_DATA_DIR) -- G lane
files /// (scripts/physics_exp + evidence/, real-data flutter-onset certification chain).

BLINDING DISCIPLINE (disclosed, not hidden -- same spirit as cert_compiler_PREREG.json's own disclosure):
the structural facts below were obtained via a FIREWALLED subagent read (a separate agent read the /
//.py +.json files and was explicitly instructed to report ONLY method/structure -- what
each stage computes and how many independent channels/producers exist -- and to omit ALL pass/fail
verdicts, headline text, correlation/n_eff numbers, and outcome-flavored language). This author (the one
writing this pipeline description and freezing predictions) has NOT read ///'s evidence
JSON "gates"/"headline"/"claims" fields, and has NOT read the.py files' own docstrings/print statements
past the firewalled summary. HONEST CONTAMINATION DISCLOSURE: this pool's naming convention bakes each
file's own finding into its FILENAME (e.g. "...beats_g758_damping_floor", "...hardens_g759"), and
directory listings (`find`) were used to SELECT this candidate before the firewall was engaged -- so the
filenames themselves were seen. Per-claim contamination is flagged below (claim 2 is HIGHER-contamination
because 's filename ~names~ its own fix "multichannel...overdet"; claim 1, the -vs- CROSS-
CHECK claim, is the cleaner, LOWER-contamination claim -- no filename states whether  and  are
ever compared against EACH OTHER, only that  is compared against  (its own robustness audit) and
that  hardens  internally).

THIS DESCRIPTION IS THE PRE-FIX STATE (mirrors cert_compiler_pipelines.py's PIPE_DCOMPOSE convention):
only 's and the OWN native facets are declared, exactly as they'd exist before  was hand-built.
 itself is treated purely as GROUND TRUTH TO BE SCORED, not encoded here.
"""

PIPE_PAZY_FLUTTER = {
    "pipeline_id": "pazy_flutter_onset_cert_G_lane",
    "argument_kind": None,  # frequency-domain FRF quantities -- no point-set/rotation-group structure;
                             # firewalled recon confirmed NO discrete symmetry (reflection/chirality) and
                             # NO explicit linear forward-measurement operator (no blur/decimate/projection
                             # matrix) anywhere in /// -- so neither the discrete_parity
                             # branch nor the linear_forward_models branch is exercised by this pipeline;
                             # this is a pure test of the shared_dof_continuous + domain_projection rules
                             # on a substrate (aeroelastic FRF) neither rule was designed against.
    "ambient_gauge": [
        {
            # the physical DOF this deployment ultimately wants certified: the real airspeed at which
            # flutter onsets (equivalently, the flutter margin from a tested airspeed). estimates it
            # via a damping-ratio linear extrapolation to zero (a functional of the HALF-POWER BANDWIDTH
            # method applied to the FRF magnitude spectrum). estimates a DECORRELATED proxy for the
            # SAME onset event via modal-frequency-COALESCENCE (the gap Delta-f between two tracked modal
            # peaks narrowing as onset approaches) -- reloaded independently from the SAME raw.mat
            # archive via its own peak-picking functional, NOT fed 's computed numbers (confirmed by
            # firewalled recon: "it does not import 's computed numbers/JSON; it independently
            # reloads the raw FRF files... via its own peak-picking function"). Two INDEPENDENT stages,
            # two DIFFERENT functionals of the same underlying physical event, same as dcompose's
            # stage2(distance_ratio)/stage3(distance_ratio) both targeting geometric_scale.
            "tag": "flutter_onset_speed", "kind": "shared_dof_continuous",
        },
        {
            # SEPARATE domain fact (declared, not derived): the Pazy rig instruments >=4 response
            # channels (x_FRF_x, x_FRF_z, z_FRF_x, z_FRF_z per firewalled recon), but 's own
            # coalescence signal, in the pre- state, is computed from exactly ONE of them
            # (x_FRF_x) -- a single-channel/single-view style domain-projection gap, structurally
            # identical in KIND to dcompose_occlusion's single-exterior-view gap (PIPE_DCOMPOSE,
            # tag=occlusion_interior), just instantiated on a sensor-channel axis instead of a
            # camera-viewpoint axis.
            "tag": "flutter_margin_channel_coverage", "kind": "domain_projection",
            "channels": ["g759_channel_x_FRF_x"],
        },
    ],
    "stages": [
        {
            "name": "g758_damping_amplitude_leg",
            "cert_facets": [
                # no ONTOLOGY entry cleanly describes "linear extrapolation-to-zero of a half-power-
                # bandwidth damping fit" -- following PIPE_OPTICS's own precedent (stageA_kk_causality's
                # Fresnel-bridge F0 estimate, "treated as a ratio-like real-valued estimate for compiler
                # purposes"), this facet is declared distance_ratio: a real-valued, independently-
                # computed estimate of the shared DOF, not a claim that it is literally a ratio of two
                # Euclidean lengths. (Named honestly as an ontology-coverage gap in the PREREG, not
                # silently forced.)
                {"cert_type": "distance_ratio", "carries": ["flutter_onset_speed"]},
            ],
        },
        {
            "name": "g759_coalescence_leg",
            "cert_facets": [
                {"cert_type": "distance_ratio", "carries": ["flutter_onset_speed"]},
                # 's coalescence facet is the SOLE (pre-) carrier of the channel-coverage tag --
                # no facet list entry is needed for the domain_projection rule (it reads "channels"
                # directly off the ambient_gauge dict, per cert_compiler.py's analyze), but the fact
                # that g759_coalescence_leg is the ONLY stage touching flutter_margin_channel_coverage is
                # what the "channels": ["g759_channel_x_FRF_x"] declaration above encodes.
            ],
        },
    ],
}

ALL_PIPELINES_PROSPECTIVE = {"pazy_flutter": PIPE_PAZY_FLUTTER}
