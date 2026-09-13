"""decorrelation_channel_cert -- the COMPOSED decorrelation-stack cert for a candidate leg-pair, realizing the agent pool-spine
requirement: deliver certs as decorrelated channels with a
lineage field). The stack has THREE layers, each catching a common-mode class the others MISS:
  LAYER-1 lineage (D d_leg_decorrelation_lineage_gate): shared backbone weights -> one channel.
  LAYER-2 error-corr on the COMPETENT region (D compose_decorrelation_verdict consumes the  evidence-gate verdict):
          measured error correlation where both legs produce output.
  LAYER-3 COMPETENCE (this lane, /477/478/479 competence_decorrelation_gate, multi-axis + worst-condition + CI-lower):
          shared competence-determining input axis -> JOINTLY BLIND where that axis is adverse, INVISIBLE to layers 1-2
          (error-corr is measured only where both are competent; on the incompetent region both abstain).

★lineage-field ALONE is INSUFFICIENT: a pair can clear layer-1 (distinct lineage) AND layer-2 (error-uncorrelated on the
competent region) yet be JOINTLY BLIND (layer-3) -- a channel lineage+error would FALSELY book as decorrelated. This cert
composes all three weakest-link and emits the channel's full field set (lineage + error + competence-per-axis), so a
decorrelated CHANNEL carries a COMPETENCE field, not only a lineage field.

Composes the structural lineage gate (layer 1+2) with the competence gate (layer 3). numpy only, no GPU."""
from __future__ import annotations
import numpy as np

from .competence_decorrelation_gate import multi_axis_competence_gate

def _d_compose(lineage_a, lineage_b, empirical_verdict):
    """layer-1 (lineage) + layer-2 (error correlation) composed verdict; returns its dict, or a
    fail-closed stub if the lineage gate is unavailable."""
    try:
        from .leg_decorrelation_lineage_gate import compose_decorrelation_verdict
    except ImportError:
        try:
            from leg_decorrelation_lineage_gate import compose_decorrelation_verdict
        except Exception:
            return dict(verdict="UNDECIDABLE", reason="lineage gate unavailable -> fail-closed", bookable=False)
    return compose_decorrelation_verdict(lineage_a, lineage_b, empirical_verdict=empirical_verdict)


def decorrelation_channel_cert(lineage_a, lineage_b, comp_a, comp_b, condition_axes, empirical_verdict="ADMIT"):
    """Full 3-layer decorrelation-channel cert. lineage_a/lineage_b: leg lineage strings (layer-1). empirical_verdict:
    the  error-corr evidence-gate verdict on the competent region (layer-2: ADMIT/FENCE/UNDECIDABLE). comp_a/comp_b:
    per-sample competence in [0,1]; condition_axes: dict {axis: per-sample condition} (layer-3, multi-axis).

    Weakest-link: DECORRELATED-CHANNEL iff layer 1+2 is DECORRELATED AND layer-3 competence is DECORRELATED-COMPETENCE.
    Otherwise the channel is FENCED / FLAGGED and the failing layer is named. Returns dict(verdict, layer1_2, competence,
    fields) -- the channel's field set."""
    d = _d_compose(lineage_a, lineage_b, empirical_verdict)
    c = multi_axis_competence_gate(comp_a, comp_b, condition_axes)
    d_ok = str(d.get("verdict", "")).upper() == "DECORRELATED"
    c_ok = c["verdict"] == "DECORRELATED-COMPETENCE"
    fields = dict(lineage_a=lineage_a, lineage_b=lineage_b, layer1_2_verdict=d.get("verdict"),
                  competence_verdict=c["verdict"], competence_per_axis=c["per_axis"],
                  competence_tested_axes=c["tested_axes"], competence_flagged_axes=c["flagged_axes"])
    if d_ok and c_ok:
        return dict(verdict="DECORRELATED-CHANNEL", layer1_2=d.get("verdict"), competence=c["verdict"], fields=fields,
                    reason="clears ALL layers: distinct lineage (1) + error-uncorrelated on the competent region (2) + "
                           "competence-decorrelated on all tested axes (3). Bookable as a decorrelated channel.")
    if not d_ok:
        return dict(verdict="FENCED-LAYER-1-2", layer1_2=d.get("verdict"), competence=c["verdict"], fields=fields,
                    reason=f"layer 1-2 (lineage/error-corr) did not admit: {d.get('reason', d.get('verdict'))}.")
    # layer 1-2 admits but competence fails -> the case lineage-field-ALONE would falsely book
    return dict(verdict="FENCED-LAYER-3-COMPETENCE", layer1_2=d.get("verdict"), competence=c["verdict"], fields=fields,
                reason=f"★clears lineage+error-corr (layers 1-2) BUT competence layer-3 = {c['verdict']} on axes "
                       f"{c['flagged_axes']}: JOINTLY BLIND where they share a competence axis -- a channel lineage+error "
                       f"ALONE would FALSELY book as decorrelated. The channel needs a COMPETENCE field, not only lineage.")


def selftest():
    rng = np.random.default_rng(0); n = 2000
    tex = rng.random(n)
    sfm = (tex > 0.4).astype(float)               # SfM competent on textured
    neural = (tex > 0.35).astype(float)           # neural also texture-gated -> jointly blind on uniform
    active = (rng.random(n) < 0.95).astype(float)  # texture-independent
    checks = []
    # (1) distinct lineage + error ADMIT + competence JOINTLY-BLIND -> FENCED-LAYER-3 (the key case)
    r1 = decorrelation_channel_cert("classical-2view-sfm-orb", "metric3d-dinov2", sfm, neural, {"texture": tex}, "ADMIT")
    checks.append(("distinct lineage + error ADMIT but competence jointly-blind -> FENCED-LAYER-3-COMPETENCE (lineage-alone "
                   "would falsely book)", r1["verdict"] == "FENCED-LAYER-3-COMPETENCE"))
    # (2) distinct lineage + error ADMIT + competence DECORRELATED -> DECORRELATED-CHANNEL
    r2 = decorrelation_channel_cert("classical-2view-sfm-orb", "metric3d-dinov2", sfm, active, {"texture": tex}, "ADMIT")
    checks.append(("distinct lineage + error ADMIT + competence decorrelated -> DECORRELATED-CHANNEL (all 3 layers pass)",
                   r2["verdict"] == "DECORRELATED-CHANNEL"))
    # (3) shared lineage -> FENCED at layer 1-2 regardless of competence
    r3 = decorrelation_channel_cert("metric3d-dinov2", "depth-anything-dinov2", sfm, active, {"texture": tex}, "ADMIT")
    checks.append(("shared DINOv2 lineage -> FENCED-LAYER-1-2 (layer-1 fences before competence matters)",
                   r3["verdict"] == "FENCED-LAYER-1-2"))
    print("decorrelation_channel_cert selftest:")
    for nm, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", nm))
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
