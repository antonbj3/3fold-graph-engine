# Running the tests

```bash
python3 -m venv .venv && . .venv/bin/activate     # any Python 3.10+
pip install -r requirements.txt                   # numpy, scipy, pyyaml, pytest, networkx, scikit-learn, pillow
pytest                                            # 66 tests: every module's own selftest (1 network test skipped)
python3 src/graph_engine/tools/tool_find.py --selftest          # search gate, synthetic corpus
python3 src/graph_engine/tools/preflight.py --graph examples/mini_graph/ANCHOR_GRAPH.json --ledger examples/mini_graph/FOLD_LEDGER.jsonl
```

Every Layer A module is also runnable on its own (`python3 src/graph_engine/cert_decorrelation.py`)
and prints its assertions. The optical worked example under `examples/optical_constants/` runs in
about a minute: `python3 kk_cert_run.py && python3 kk_decorrelation_run.py && python3
qc_neff_sign_blindness.py` (run from that directory). It is a NEGATIVE result about a negative
result: the pooled "the two checks are co-blind" reading is overturned by the QC script, because
pooling across material classes with |rho| hides the sign of the correlation.

## Environment variables

Every path that was absolute in the source project is a parameter here. Defaults are relative to
the current working directory.

| variable | used by | default |
|---|---|---|
| `ANCHOR_GRAPH`, `FOLD_LEDGER`, `GRAPH_STORE` | `tools/anchor_graph_tools.py`, `tools/fold_ledger_tools.py`, `tools/graph_store.py` | `./ANCHOR_GRAPH.json`, `./FOLD_LEDGER.jsonl`, `./GRAPH_STORE.json` |
| `TOOL_FIND_ROOTS` | `tools/tool_find.py` | current directory |
| `ALLINDEX_ROOTS`, `ALLINDEX_DB`, `ALLINDEX_STATS` | `tools/allindex.py` | `cwd=.`, `data/allindex/allindex.db` |
| `DATA_GRAPH` | `tools/data_graph_tools.py` | `./DATA_GRAPH.json` |
| `CONTEXT_CORPUS_DIR`, `CONTEXT_CORPUS_DIR_2` | `tools/context_loader.py` | `./corpus` |
| `DATASET_PROFILER_BASE`, `DATA_SOURCES_DB`, `RETRODICT_DIR`, `RETRODICT_SCHEMA` | `tools/dataset_profiler.py` | cwd, `./data_sources.db` |
| `CERT_ARTIFACT_ROOT`, `CERT_GRADE_CLAUSE_DIR` | `cert_grade_engine.py` | cwd, unset |
| `CERT_EVIDENCE_DIR`, `PIPELINE_SOURCE_ROOT`, `PAZY_DATA_DIR` | `cert_compiler*.py` | `./evidence`, `.` |
| `EXTERNAL_SRC_PATTERN` | `overdet_provenance_check.py` | `[\w.-]+-[A-Z]/src` |
| `DECORR_RECOVERY_IMAGE_DIR` | `decorrelated_recovery.py` | unset (synthetic textures) |
| `MERL_BRDF_DIR`, `OED_PRIOR_RUN_JSON`, `OED_STOPPING_RULE_OUT` | `examples/oed_probes/` | relative |
| `FOLD_GATE_OUT`, `ATOMS_VALIDATION_OUT` | `tools/fold_gate*.py`, `tools/atoms_validation.py` | `./reports/probes` |
| `NOTES_INDEX`, `GRAPH_DOC`, `LEDGER_DOC` | `tools/system_map.py` | `docs/*.md` |
| `PIPELINE_PYTHON` | `paper_graph/pipeline.py` | the running interpreter |
| `PAPER_CORPUS`, `PAPER_BUNDLE` | `paper_graph/pipeline.py --real` | `corpus.jsonl`, `paper_graph.pkl` next to the module |
| `PAPER_GRAPH_NETWORK` | `tests/test_paper_graph_example.py` | unset (the fetch test is skipped) |

Two modules need a component this repo does not ship and fail closed without it (by design):
`tools/dataset_profiler.py` requires a `dataset_band_port` module exposing `apply_band(profile)`
before it will serve a profile (its selftest passes an identity band), and `cert_grade_engine.py`
skips the optional C7-C11 clauses unless `$CERT_GRADE_CLAUSE_DIR` points at them.

## The paper_graph example (public arXiv set)

`examples/paper_graph/arxiv_ids.txt` freezes 50 arXiv identifiers for the topic
"topology optimization" (one of `download_papers.py`'s own `DEFAULT_TOPICS`). arXiv metadata
(titles, abstracts, authors, categories) is redistributable and is what the pipeline consumes,
so `examples/paper_graph/corpus.jsonl` (50 records, 112 kB) is shipped and the offline tests read
it. Full PDFs are never shipped; Semantic Scholar enrichment (citation counts + reference ids) is
fetched at run time and git-ignored. Regenerate everything with:

```bash
python3 src/graph_engine/paper_graph/download_papers.py --ids-file examples/paper_graph/arxiv_ids.txt \
        --out examples/paper_graph/corpus.jsonl --reset                    # metadata only, shipped
python3 src/graph_engine/paper_graph/download_papers.py --ids-file examples/paper_graph/arxiv_ids.txt \
        --out examples/paper_graph/corpus_s2.jsonl --reset --s2            # + citation edges, not shipped
python3 src/graph_engine/paper_graph/build_paper_graph.py --corpus examples/paper_graph/corpus.jsonl \
        --out examples/paper_graph/paper_graph
python3 examples/paper_graph/run_paper_graph_example.py
PAPER_CORPUS=examples/paper_graph/corpus_s2.jsonl PAPER_BUNDLE=examples/paper_graph/paper_graph_s2.pkl \
        python3 src/graph_engine/paper_graph/pipeline.py --real
```

Numbers behind the three VERIFIED-FRESH rows:

* `download_papers.py` — 50 of 50 requested ids returned a record, every record carries a title,
  abstract and author list; offline `--self-test` green.
* `build_paper_graph.py` — metadata-only corpus: 50 nodes, 74 edges (all similarity), density
  0.0604, 6 components (largest 40, 1 singleton), 27 triangles. With the enrichment run:
  50 nodes, 78 edges (60 similarity, 14 both, 4 citation), 18 directed citation links inside the
  set, 2228 dangling references, 4 components (largest 42), 29 triangles.
* `pipeline.py --real` — on the enriched bundle: 31 distinct real papers retrieved and certified,
  verdicts all ABSTAIN (rungs C-reach 24, A-gauge 2, B-weak 5), which is the documented correct
  output for abstract-only evidence; gates R1-R6 all PASS, `ALL_PASS = True`.

The other five modules stay SYNTHETIC-ONLY, each with the gate that this set cannot satisfy:

* `paper_graph_search.py` — its own falsifier: "On the REAL paper-graph, if rho(struct, citation)
  >= 0.7 OR rho(struct, semantic) >= 0.7, the method REDUCES to that baseline there." Measured
  here: rho(struct, semantic) = -0.777 (enriched) / -0.773 (metadata only), i.e. |rho| = 0.78
  above the 0.7 ceiling, so on this single-topic set the structural composite is rank-redundant
  with the semantic baseline up to sign. rho(struct, citation) = -0.252 (undefined without the
  enrichment, since arXiv metadata carries no citation counts). Numbers in
  `examples/paper_graph/paper_graph_search_eval.json`.
* `eval_search_vs_baselines.py` — its pre-registration requires planted ground truth ("Ground
  truth = the A-B hole bridges", with the hardest distractors generated to be content-identical);
  a public arXiv set carries no such labels, so MAP against the four fair baselines is not
  computable on it.
* `scale_index.py` — its gate is the scaling exponent and LSH recall against the exact all-pairs
  oracle on "a synthetic ~2000-paper corpus"; 50 papers do not reach that regime.
* `search_cert_abstain.py` — its own `main()` gates run the hand-built ground-truth cases
  (`build_cases`, each case carries a "should" verdict). The real set exercises it only through
  `pipeline.py --real` PART A, where the honest output is ABSTAIN for every paper.
* `unified_graph_engine.py` — a compat re-export of `graph_hole_engine`; its demo runs on the
  adapter's own four-paper corpus and takes no external corpus.

`tests/test_paper_graph_example.py` runs the build + overlay + search on the shipped metadata
cache offline; the fetch test is skipped unless `PAPER_GRAPH_NETWORK=1`.

## Status

VERIFIED-FRESH = selftest passed in this repo's venv, numbers reproduced · CUDA-ONLY = not runnable
here until the GPU driver is fixed; last verified in the source project · SYNTHETIC-ONLY = method
demonstrated in the source project on private data; ships with synthetic input and a test, real-data
result not reproducible from this repo.

| module | status |
|---|---|
| `examples/inverse_design/persona_design_generative_beats_patent.py` | VERIFIED-FRESH |
| `examples/inverse_design/persona_design_holdout_exam.py` | VERIFIED-FRESH |
| `examples/inverse_design/persona_design_inverse_classified.py` | VERIFIED-FRESH |
| `examples/inverse_design/persona_design_multipatent_cluster.py` | VERIFIED-FRESH |
| `examples/inverse_design/spine_regression_suite.py` | VERIFIED-FRESH |
| `examples/oed_probes/a7_recalibrate_oed_stopping_rule.py` | VERIFIED-FRESH |
| `examples/oed_probes/fleet_oed_vibrate_the_fleet_read_collective_soft_mode_and_next_best_acquisition.py` | VERIFIED-FRESH |
| `examples/oed_probes/multimodal_dither_lifts_null_space_active_two_kinds_acquisition_fisher_adds.py` | VERIFIED-FRESH |
| `examples/oed_probes/vibrating_multitone_OED_bundle_FDT_lockin_reads_fisher_spectrum_model_free.py` | VERIFIED-FRESH |
| `examples/optical_constants/kk_cert_lib.py` | VERIFIED-FRESH |
| `examples/optical_constants/kk_cert_run.py` | VERIFIED-FRESH |
| `examples/optical_constants/kk_decorrelation_lib.py` | VERIFIED-FRESH |
| `examples/optical_constants/kk_decorrelation_run.py` | VERIFIED-FRESH |
| `examples/optical_constants/kramers_kronig_passivity_cert.py` | VERIFIED-FRESH |
| `examples/optical_constants/qc_neff_sign_blindness.py` | VERIFIED-FRESH |
| `src/graph_engine/admission_bands.py` | VERIFIED-FRESH |
| `src/graph_engine/agreement_admission_gate.py` | VERIFIED-FRESH |
| `src/graph_engine/black_box_sigma_min_detector.py` | VERIFIED-FRESH |
| `src/graph_engine/cert_compiler.py` | SYNTHETIC-ONLY |
| `src/graph_engine/cert_compiler_pipelines.py` | SYNTHETIC-ONLY |
| `src/graph_engine/cert_compiler_prospective_pazy_pipeline.py` | SYNTHETIC-ONLY |
| `src/graph_engine/cert_decorrelation.py` | VERIFIED-FRESH |
| `src/graph_engine/cert_federation_measure.py` | SYNTHETIC-ONLY |
| `src/graph_engine/cert_grade_engine.py` | VERIFIED-FRESH |
| `src/graph_engine/certifying_power_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/claim_federation.py` | SYNTHETIC-ONLY |
| `src/graph_engine/classical_feature_lineage_registry.py` | VERIFIED-FRESH |
| `src/graph_engine/competence_decorrelation_gate.py` | VERIFIED-FRESH |
| `src/graph_engine/coupling_admission_precheck.py` | VERIFIED-FRESH |
| `src/graph_engine/coverage_collapse_alarm.py` | VERIFIED-FRESH |
| `src/graph_engine/coverage_strata_guard.py` | VERIFIED-FRESH |
| `src/graph_engine/decorrelated_abstain_field.py` | VERIFIED-FRESH |
| `src/graph_engine/decorrelated_probe_value.py` | VERIFIED-FRESH |
| `src/graph_engine/decorrelated_recovery.py` | SYNTHETIC-ONLY |
| `src/graph_engine/decorrelation_channel_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/decorrelation_validity_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/fleet_oed.py` | VERIFIED-FRESH |
| `src/graph_engine/fleet_oed_registry_bridge.py` | SYNTHETIC-ONLY |
| `src/graph_engine/fluctuation_oed.py` | VERIFIED-FRESH |
| `src/graph_engine/gauge_aware_admit.py` | VERIFIED-FRESH |
| `src/graph_engine/gauge_catalog.py` | VERIFIED-FRESH |
| `src/graph_engine/gauge_completeness_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/gauge_detector.py` | VERIFIED-FRESH |
| `src/graph_engine/gauge_kind_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/generative_leg_admission.py` | VERIFIED-FRESH |
| `src/graph_engine/graph_hole_engine/__main__.py` | SYNTHETIC-ONLY |
| `src/graph_engine/graph_hole_engine/adapters.py` | VERIFIED-FRESH |
| `src/graph_engine/graph_hole_engine/engine.py` | VERIFIED-FRESH |
| `src/graph_engine/graph_hole_engine/kernel.py` | VERIFIED-FRESH |
| `src/graph_engine/graph_hole_engine/oed_sibling.py` | VERIFIED-FRESH |
| `src/graph_engine/graph_hole_engine/tests.py` | VERIFIED-FRESH |
| `src/graph_engine/graph_interface.py` | SYNTHETIC-ONLY |
| `src/graph_engine/identifiability_oed.py` | VERIFIED-FRESH |
| `src/graph_engine/inverse_design/persona_design_gi_service.py` | VERIFIED-FRESH |
| `src/graph_engine/leg_admission.py` | VERIFIED-FRESH |
| `src/graph_engine/leg_decorrelation_lineage_gate.py` | VERIFIED-FRESH |
| `src/graph_engine/lens_pooling.py` | SYNTHETIC-ONLY |
| `src/graph_engine/linalg_guards.py` | VERIFIED-FRESH |
| `src/graph_engine/literature_grader_prototype.py` | VERIFIED-FRESH |
| `src/graph_engine/llm_federation_analysis.py` | SYNTHETIC-ONLY |
| `src/graph_engine/llm_federation_data.py` | SYNTHETIC-ONLY |
| `src/graph_engine/llm_federation_forced_adversary.py` | SYNTHETIC-ONLY |
| `src/graph_engine/llm_federation_gen.py` | CUDA-ONLY | generation driver; on 2026-09-12 CUDA torch + transformers were present but the Hugging Face model download failed (connection refused from this host); not run. Needs the models named in its spec table on disk. 
| `src/graph_engine/margin_net.py` | SYNTHETIC-ONLY |
| `src/graph_engine/mechanism_signature.py` | VERIFIED-FRESH |
| `src/graph_engine/member_decorrelation_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/monotone_agnostic_admission.py` | VERIFIED-FRESH |
| `src/graph_engine/motion_validity_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/neff_channel_diversity.py` | VERIFIED-FRESH |
| `src/graph_engine/neff_form.py` | VERIFIED-FRESH |
| `src/graph_engine/observability_sigma_min_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/oed_probe_selection_engine.py` | SYNTHETIC-ONLY |
| `src/graph_engine/oed_probe_selection_engine_v2_instance.py` | SYNTHETIC-ONLY |
| `src/graph_engine/overclaim_gauge.py` | VERIFIED-FRESH |
| `src/graph_engine/overdet_budget.py` | VERIFIED-FRESH |
| `src/graph_engine/overdet_neff.py` | VERIFIED-FRESH |
| `src/graph_engine/overdet_precondition_gate.py` | VERIFIED-FRESH |
| `src/graph_engine/overdet_provenance_check.py` | VERIFIED-FRESH |
| `src/graph_engine/paper_graph/build_paper_graph.py` | VERIFIED-FRESH |
| `src/graph_engine/paper_graph/download_papers.py` | VERIFIED-FRESH |
| `src/graph_engine/paper_graph/eval_search_vs_baselines.py` | SYNTHETIC-ONLY |
| `src/graph_engine/paper_graph/paper_graph_search.py` | SYNTHETIC-ONLY |
| `src/graph_engine/paper_graph/pipeline.py` | VERIFIED-FRESH |
| `src/graph_engine/paper_graph/scale_index.py` | SYNTHETIC-ONLY |
| `src/graph_engine/paper_graph/search_cert_abstain.py` | SYNTHETIC-ONLY |
| `src/graph_engine/paper_graph/unified_graph_engine.py` | SYNTHETIC-ONLY |
| `src/graph_engine/plan_value.py` | SYNTHETIC-ONLY |
| `src/graph_engine/polarity_rules.py` | VERIFIED-FRESH |
| `src/graph_engine/pooled_screening.py` | SYNTHETIC-ONLY |
| `src/graph_engine/probe_gauge_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/record_guarantee.py` | SYNTHETIC-ONLY |
| `src/graph_engine/regime_posterior.py` | SYNTHETIC-ONLY |
| `src/graph_engine/reliability_gate_decorr_check.py` | VERIFIED-FRESH |
| `src/graph_engine/render_match_decorrelation_judge.py` | VERIFIED-FRESH |
| `src/graph_engine/replay_policy.py` | VERIFIED-FRESH |
| `src/graph_engine/resistance_sketch.py` | VERIFIED-FRESH |
| `src/graph_engine/scene_eyes_coverage_audit.py` | VERIFIED-FRESH |
| `src/graph_engine/scene_eyes_neff_leg.py` | VERIFIED-FRESH |
| `src/graph_engine/stage_decorrelation_verifier.py` | VERIFIED-FRESH |
| `src/graph_engine/source_reliability.py` | SYNTHETIC-ONLY |
| `src/graph_engine/closed_loop.py` | SYNTHETIC-ONLY |
| `src/graph_engine/profile.py` | VERIFIED-FRESH |
| `src/graph_engine/representation_probe.py` | VERIFIED-FRESH |
| `src/graph_engine/precision_form.py` | VERIFIED-FRESH |
| `src/graph_engine/hidden_variable.py` | SYNTHETIC-ONLY |
| `src/graph_engine/numeric_rules.py` | VERIFIED-FRESH |
| `src/graph_engine/provenance_rules.py` | VERIFIED-FRESH |
| `src/graph_engine/regime_markov.py` | SYNTHETIC-ONLY |
| `src/graph_engine/next_actions.py` | VERIFIED-FRESH |
| `src/graph_engine/claim_types.py` | VERIFIED-FRESH |
| `src/graph_engine/decision_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/guarantees.py` | VERIFIED-FRESH |
| `src/graph_engine/disagreement_field.py` | SYNTHETIC-ONLY |
| `src/graph_engine/throws.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/allindex.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/anchor_graph_tools.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/atoms_lint.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/atoms_runner.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/atoms_validation.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/context_loader.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/context_loader_eval.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/data_graph_tools.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/dataset_profiler.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/fold_aggregator.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/fold_gate.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/fold_gate_v2.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/fold_ledger_tools.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/graph_store.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/preflight.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/system_map.py` | SYNTHETIC-ONLY |
| `src/graph_engine/tools/tool_find.py` | SYNTHETIC-ONLY |
| `src/graph_engine/typed_extraction.py` | SYNTHETIC-ONLY |
| `src/graph_engine/unlock_value.py` | SYNTHETIC-ONLY |
| `src/graph_engine/video_admission_agreement_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/video_admission_cert.py` | VERIFIED-FRESH |
| `src/graph_engine/video_admission_identifiability_pregate.py` | VERIFIED-FRESH |

A cross-repository example that writes a measurement node into this schema and loads it with `graph_store` lives in the motion engine repository: `examples/compose/field_sdf_to_contacts_to_graph.py`.
