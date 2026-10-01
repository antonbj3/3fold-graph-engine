# Research admission consumers

These are copies of the existing native graphctl and F1 consumer, changed only
in this research worktree. `state` and `BANK_LAYER.json` point to this lane's
frozen snapshots on the designated disk. All bank builder output paths have
been relocated to G5. Canonical consumers and graphs are unchanged.

The existing `fold_gate_v2_round` runs the strict a/b/c obligations. c requires
an external `verification_registry` supplied by the consumer, mapping source
IDs to `{verified: true/false, source_family: [IDs], supports: [target IDs]}`. Producer artifact labels
are not verification authority. Missing registry, unknown IDs or unsupported targets block.
A registry is a trust boundary; correct external family assignments remain an
assumption. No adjacency or documentary link confers another independent vote.

The receipt artifact carries `agreement_calibration` (frozen selection, oriented
scores, binary external labels, unique candidate IDs, source families) and a
`validation` witness (instrument/reference families, external event basis,
independent test relation). These are scoped, externally reviewable inputs,
not automatic verification of arbitrary scientific meaning. Calibration may
not be trained or selected on its own truth labels. The family bootstrap uses
exactly the agreement's k in each resampled pool.

Legacy direct gate calls remain compatible; strict admission is explicit.
`agreement_gain_credited` is false without equal-selectivity calibration.
The copied recorder requires admission submission/template/registry for
PROVEN, REFUTED or regime-gate opening, before state mutation. Ordinary artifact
recording remains unadjudicated. Template and proposal artifact paths should
be absolute. The template is a known-valid external positive control, not the
proposal's own self-certification; mutants are fresh each round.

F1 `admission_preview` evaluates bound proposal receipts without promoting any
history node. No default field says that PENDING producer history became proven.
Run `run_ablation.py`, `test_consumers.py`, `run_selftests.py` in the G5 lane;
set PYTHONPATH to this worktree's src for standalone gate imports.
