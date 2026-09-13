"""classical_feature_lineage_registry -- the SECOND common-mode axis: common-mode has TWO
axes -- shared trained-WEIGHTS (D's WEIGHTS_LINEAGE_FAMILIES, dinov2 class) AND shared classical FEATURE. Two
classical legs that BOTH compute the same feature family (e.g. two HSV colour-histogram legs, or two block-colour
legs) share a FEATURE common-mode -- their errors correlate on the SAME nuisance (lighting/colour) even though
neither has trained weights. The weights-registry is BLIND to this (2 HSV legs get the SAME verdict as
HSV+temporal). This registry is the FEATURE-axis content, designed to feed C's axis-agnostic
`ecc_admission_precondition(..., verified_lineage_registry=)` resolution: declared leg-ids
resolve to a canonical FEATURE FAMILY, and two legs sharing a family are fenced exactly like shared weights.

★HONEST SCOPE (same as the weights-lineage fence): a feature-family assignment
is a DECLARED a-priori taxonomy grounded on what each leg actually computes -- it is a LAYER-1 FLAG (a cheap prior
that the score n_eff can miss), NOT a determinant. The EMPIRICAL n_eff still determines;
compose_decorrelation_verdict is the canonical fused (lineage x empirical) tool. Two legs in the same family CAN
be empirically decorrelated on a sample; the fence says "shared feature -> correlated ERRORS the scores may hide",
report both and fuse weakest-link. Grounded on C's real classical appearance legs (src/graph_engine):
  * appearance_selfconsistency_leg: GxG grid of per-block (mean-colour, gradient-energy) -> 'block-colour-gradient'
  * point_patch_appearance_leg: patch_structure_residual (local structure) -> 'patch-structure'
  * photo_ncc: raw-intensity normalized cross-correlation -> 'raw-intensity-ncc'
  * HSV colour-histogram legs (football track-id) -> 'colour-histogram'
  * sensor noise-residual / PRNU -> 'sensor-noise-residual'
"""

# declared leg feature-id -> canonical FEATURE FAMILY. Two legs mapping to the same value share a feature common-mode.
CLASSICAL_FEATURE_FAMILIES = {
    # colour-histogram family (global HSV/colour distribution) -- the 2-HSV-legs case
    "hsv_histogram": "colour-histogram",
    "hsv_track": "colour-histogram",
    "hsv_appearance": "colour-histogram",
    "colour_histogram": "colour-histogram",
    # block-colour-gradient family (per-block spatial mean-colour + gradient-energy)
    "appearance_selfconsistency": "block-colour-gradient",
    "block_colour_z": "block-colour-gradient",
    # patch-structure family (local structural appearance)
    "point_patch_structure": "patch-structure",
    "patch_structure_residual": "patch-structure",
    # raw-intensity NCC family
    "photo_ncc": "raw-intensity-ncc",
    # sensor-noise-residual / PRNU family
    "noise_residual": "sensor-noise-residual",
    "prnu": "sensor-noise-residual",
}


def feature_family(leg_id):
    """Canonical feature family for a declared classical leg-id, or the id itself if unknown (unverified -> the
    caller's ecc gate reports lineage_verified=False for that leg, honest per the resolution contract)."""
    return CLASSICAL_FEATURE_FAMILIES.get(leg_id, leg_id)


def merged_lineage_registry(weights_families=None):
    """The TWO-AXIS registry: classical FEATURE families UNION shared-WEIGHTS families. Feed the result to
    ecc_admission_precondition(..., verified_lineage_registry=) to fence BOTH a shared-weights pair (dinov2) AND a
    shared-feature classical pair (2 HSV) with one axis-agnostic resolution -- neither axis alone covers both.
    weights_families: an optional {declared_id -> canonical_weights_family} map (e.g. D's WEIGHTS_LINEAGE_FAMILIES /
    a leg->backbone map); merged so a canonical collision on EITHER axis fences."""
    merged = dict(CLASSICAL_FEATURE_FAMILIES)
    if weights_families:
        merged.update(weights_families)
    return merged
