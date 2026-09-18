# Engine additions: contract, what each module computes, and what was measured

Every number below is produced by a script in this directory (`python3 examples/engine_experiments/<script>.py`;
seeds fixed; result JSON next to the script). e1, e3, e3b, e3c, e6, e8, e8b download the public SNAP graphs
cit-HepTh / cit-HepPh to `$HUNT_DATA`. e7 needs a local copy of Qwen2.5-0.5B-Instruct (`$JUDGE_MODEL`); its
measured answers are shipped as `e7_probabilities.npy`, so e7b, e9, e10 run without the model.
Test graphs with planted truth are labelled TEST; `curated_two_graphs.json` is a hand-written fixture from public
textbook relations. No private graph is included or was used for any number in this file.

## Contract (what data has to look like)

A graph is a JSON object. All four record types share `sources`, the lineage that decides what counts as independent.

```
sources   [{"id": "Hall1951", "derives_from": []}, {"id": "Textbook", "derives_from": ["Hall1951", "Petch1953"]}]
claims    [{"id", "subject", "object", "sign": +1|-1, "validity": {"grain_size_nm": [25, 1e5]}, "evidence": [source ids], "cost"}]
edges     [{"id", "between": [var, var], "reports": [{"margin": 0.12, "sigma": 0.05, "sources": [ids], "validity": {...}}], "weight", "cost"}]
scales    {"grain_size_nm": "log"}                       optional; default linear
nodes     the existing ANCHOR_GRAPH node: id, claim, type, status, depends_on, risk, cost, evidence
```
`margin` = (capacity − demand) / scale, negative = violated. `sigma` is its standard deviation; without it
`MarginNet.default_sigma` is used for every report, and rankings should then be checked over a range of values.

## Relation to the document pipeline already in this repository

`src/graph_engine/paper_graph/pipeline.py` is the existing design for getting documents into the graph:
download → paper graph with an extracted `Claim {domain, mechanism atoms, scope, polarity, lineage, provenance}` → search →
grade against the target node → C_pred update (close hole / add confirmer / flag refutation), with an ABSTAIN gate at
`extraction_confidence < 0.6`. Its own register marks the prose → structured-claim extraction as STUBBED, and on the shipped
corpus the gate abstains on all 57 papers from title and abstract. The modules below do not replace that pipeline.
- Two claim forms now exist. `Claim` (pipeline) states a capability: mechanism atoms under a scope. `claims` (contract above)
  state a relation between two variables. polarity ↔ sign, scope ↔ validity, lineage/provenance ↔ sources.
- One independence rule: `confidence_from_legs` counts distinct lineage keys; `claim_federation.n_eff` gives the same number
  when each source has one origin and extends it to a lineage tree (test).
- `typed_extraction` is that stubbed step. It returns a `pipeline.PaperNode`, so the existing grader, the ABSTAIN gate at 0.6 and
  `kg_update` run unchanged (test). It needs a judge; none is shipped. e12 uses a simulated one, e7 measured a local 0.5B model.
- Which N_eff. The repository keeps several, by purpose (`neff_form.py`; the NEFF_BAR note in `leg_decorrelation_lineage_gate.py`):
  Kish (ρ) for variance reduction, participation ratio (ρ²) for decorrelation rank, 2/(1+ρ) for a false-accept floor.
  `claim_federation.lineage_information` = ‖M⁺1‖² is in the variance-reduction family: Kish × the information of one report, with
  GLS weights (test against `neff_form`). It covers the structural layer only; the measured-correlation layer is not composed.

## Modules

| module | question it answers | rule |
|---|---|---|
| `resistance_sketch` | hole field L⁺_ii and resistance R_ij without a dense inverse | L⁺ = E[ZZᵀ], Z from k sparse solves; L⁺_ii = (1 − 1/n + Σ_j w_ij L⁺_ij)/d_i keeps the local term exact; 4-bit rotated codes |
| `claim_federation` | two graphs, shared concepts: support, contradiction or regime boundary, inferred links | N_eff = ‖M⁺1‖² over root sources; opposite signs + intersecting boxes = contradiction, disjoint = boundary; inferred links keep both leg ids and never enter the claim set |
| `regime_posterior` | all claims on one pair read jointly; where to probe | exact Bayes over ≤1 transition on a FIXED partition, with the two-transition family kept at prior 0.05; probe value = expected drop of the potential, never negative |
| `margin_net` | requirements with margins: stress, disagreement, next measurement, shared-source risk | pseudo-inverse GLS with Σ = D M Mᵀ D (same N_eff); z = m̂/s; χ² disagreement plus a separate check that declared copies agree; pre-posterior entropy drop; failure-count distribution with shared roots |
| `unlock_value` | order of work when open nodes can fail | largest-ratio initial chain segment first, ratio = (1 − Πp)/expected cost; optimal for one goal with no or chain precedence (brute force over every feasible order, 160 graphs); heuristic for trees and for several goals |
| `replay_policy` | which pairs get connected next, learned from the graph's own history | logistic policy on graph features; `grow_span` adds the direction most correlated with the residual; optional 1/π weights |
| `throws` | new links to try | far in graph, far in subject, close in mechanism; systematic sampling with EXACT inclusion probabilities (test against 6 000 draws); pair-list form for large graphs |
| `mechanism_signature` | are two models mechanically alike | limit points, tangency order, β, γ, oddness of F(x, μ) = 0; `signature_vector` does the same for x ∈ ℝⁿ by pseudo-arclength continuation (limit point = tangent's μ-component changes sign, σ_min(∂F/∂x) → 0) |
| `graph_interface` | combine two graphs through shared concepts only | R_ab = g_a + g_b + (h_a − h_b)ᵀ(S_A + S_B)⁺(h_a − h_b); exact, with g exact by default (a Hutchinson estimate only on request) |
| `lens_pooling` | pool several phrasings of one question to one judge | per-stratum Platt scaling, error correlation, tempering, balanced lens sets from a Hadamard array |
| `pooled_screening` | how many candidates per pooled question | two-stage Dorfman with noisy tests, closed forms |
| `plan_value` | probing and working in one currency | settle cost per node by a dynamic programme over instruments (cost, reliability) with admission level τ; nodes ordered by (1 − p)/settle cost; cheap-instrument value = S_expensive − S_all ≥ 0 |
| `record_guarantee` | admit extracted records so that ≥ 1 − α of the admitted are entirely right, distribution-free | score = least certain field; threshold = largest of 20 quantile candidates whose Clopper–Pearson (1 − δ/20) bound is ≤ α (Learn-then-Test); assumes exchangeability only |
| `source_reliability` | how often each independent origin is right, from agreement alone | Dawid–Skene EM with a Beta prior over roots (copies collapsed first); feeds `Federation(reliability_by_root=…)` |
| `polarity_rules` | the sign a sentence asserts between two quantities, symbolically | one direction word per quantity per clause, negation flips, last clause wins, composition by product; abstains outside its lexicon (English only) |
| `typed_extraction` | the stubbed prose → `Claim` step of `paper_graph/pipeline.py` | yes/no fields from the target node's claim; balanced lenses; isotonic calibration of the pooled log-odds; confidence = Π max(p, 1−p) = P(whole claim right); `agree` adds the log-odds of a second, independent judge |

## Measured

**Speed and memory (e1).** Dense pinv against sketch k = 64 + 4-bit codes. n = 4 000: 61.4 s → 1.5 s, 128 MB → 0.14 MB.
cit-HepTh, 27 400 nodes: dense not run (6.0 GB); sketch 2.3 s. Three estimates, not to be mixed: float sketch (14.0 MB) rank
correlation with exact values 0.9887, median relative error 0.119; 4-bit codes (0.99 MB) 0.9883, 0.112; refined (needs the float
sketch and the sparse Laplacian) 0.9993, 0.022. Over all 11 rows the 4-bit codes lower the rank correlation by 0.0026 on average
(at most 0.0084) and raise the median relative error by 0.0017 on average (at most 0.0065). Sparse LU on the citation
graph gave no result in 10 min; block CG is used above 6 000 nodes. The SIMD scan kernels of turbovec are not reproduced.

**Two graphs (e2, TEST, 8 seeds × 4 noise settings).** Sign accuracy 0.79 → 0.86 and inferred-link precision 0.23 → 0.71 with
validity boxes; lineage adds 0.013–0.014 when 30–60 % of sources are copies, 0 otherwise. Boundary recall 1.0, precision 0.47–0.64.

**Joint reading (e4, TEST).** Pointwise vote 0.653 → regime posterior 0.779 with no probe (other settings: 0.590 → 0.677,
0.644 → 0.762, 0.649 → 0.782). After 60 probes: random 0.807, greedy with the entropy potential 0.839, greedy with min(p,1−p)
0.845, pointwise vote with its own ranking 0.683 (120 answers). min(p,1−p) scored 0.001–0.006 above entropy in all four
settings; entropy remains the default because its value is positive for every informative probe and has the Fisher-coordinate
reading, not because it scored higher. Probe value is never negative and equals the realized expectation (test). A first
version let probes add cells to the hypothesis space; there the value could be negative and mispredicted (review), and two
statements about probe placement made from it were artifacts and are withdrawn in the module docstring.
Greedy two-probe plan ≥ (1−1/e) × best adaptive plan: checked by brute force for near-noiseless answers only.

**Collision family (e4b, TEST).** Two transitions cannot be detected passively at 1–3 claims per pair (recall 0). With probes on
one pair: recall 0.25 / 0.89 / 0.99 at 4 / 8 / 16 probes, false flags 0.007 / 0.007 / 0.000. With prior 0 the family can never be
seen (test).

**Replay policy (e3, e3c; public SNAP graphs, label = pair first co-cited within 24 months).** AP, HepTh@95 / HepPh@95: base rate
0.034 / 0.052; existing hole field 0.025 / 0.036; best single feature 0.123 / 0.142; replay policy 0.167 / 0.196; fitted on the
other graph only 0.164 / 0.195; after 8 residual rounds 0.189 / 0.210 (8 random directions: 0.176 / 0.204); the gain is 1.8–2.3 times the random-direction gain
over the four graph × cutoff cells. Same pattern at
cutoff 107. No gain: the log-linear extrapolation feature (Δ ≤ 0.001), resistance features (≤ 0.003), choosing replay subgraphs
where the policy is worst (−0.008 on HepPh@95). e3b: among pairs that do get connected, larger resistance distance goes with a
LOWER citation percentile of the connecting paper (Spearman −0.03 to −0.20, six of six cells).

**Throws with outcomes known only for throws made (e8, e8b).** Links found / AP of the final policy on an unseen period (HepTh):
top-K 3 608 / 0.130; 30 % random 2 795 / 0.137; softmax T = 1 1 417 / 0.161; uniform 527 / 0.162; softmax T = 0.5 with 1/π-weighted
fit 2 491 / 0.163 — the same AP as uniform within the seed spread (sd 0.002–0.005), with 4.7 times the links (exact inclusion probabilities, systematic sampling; HepPh: 1 463 / 0.190 against uniform 497 / 0.192). Reference with all labels: 0.167.

**Cheap judge plus expensive probe at equal spend (e5, SIMULATION, judge parameters assumed).** Expensive only 0.845; judge sweep
then expensive probes, answers tempered by N_eff: 0.877–0.971 over five settings. Treating correlated judge answers as independent
is worse than no probes at error length 0.4 (accuracy 0.802 vs 0.779 but Brier 0.170 vs 0.153); tempered: Brier 0.115.

**Throw screening as a cascade (e6; stage 1 = replay policy on real SNAP data, stage 2 = SIMULATED judge with d′, stage 3 = verification
at 200× the judge's cost; budget = 1 000 verifications).** Real links confirmed, HepTh / HepPh: verify at random 33 / 56; stage 1 then
verify 406 / 466; stage 1 → judge → verify 636 / 700 (d′ = 1), 894 / 910 (d′ = 2), 964 / 971 (d′ = 3). Judging all 400 000 candidates
costs more than the whole budget.

**A local 0.5B judge, measured (e7, e7b, e9, e10; 320 labelled TEST sentences, 4 phrasings × 2 option orders).** Repeat of one lens:
identical (max diff 0.0). Error correlation same option order +0.61, swapped −0.08. Option-position bias ±0.7–0.9 log-odds. Single
lens 0.54; four lenses one order 0.533; four balanced on order 0.582; all eight 0.591. Per-(sentence form, lens) Platt scaling,
fitted on 10 quantity pairs and applied to the other 10: single 0.838, pooled 0.927 (20 splits, sd 0.025). No gain: asking direction
of X and of Y separately and multiplying (e7c: 0.572); none of PR, Kish, 1ᵀC⁻¹1 predicts the pooling gain of 3- and 4-lens subsets
(Spearman −0.24 to 0.09). Tempering did not improve Brier here (0.055 → 0.059).

**Order of work (e11, SIMULATION).** Expected spend relative to the optimal order: existing `priority` 1.207; new rule with the
LOW/MED/HIGH class already stored in the graph 1.025; with p mis-estimated by logit noise 0.5 / 1.0 / 2.0: 1.035 / 1.117 / 1.289 —
at noise 2.0 the new rule is WORSE than the existing one;
random 2.093. Nodes sharing a cause of failure (probability 0.1 / 0.3, rule applied with marginal p, reference = brute-force
optimum under the joint law, 6 nodes): new rule 1.005 / 1.026, existing `priority` 1.100 / 1.114.

**Documents to graph through the existing pipeline (e12, SIMULATION; judge with order bias ±0.8 log-odds, lens noise, and a
misreading shared by all lenses with probability q; 43 papers, 27 graph writes due; 200 seeds).** Harmful writes / correct writes:
q = 0: one lens 0.02 / 7.6 (abstains on 45 %); four lenses of one option order 0.51 / 20.9; four balanced, tempered 0.02 / 26.8.
q = 0.1: four balanced, tempered 3.9 / 21.0; the same isotonically calibrated 3.3 / 17.2; two independent judges, calibrated, log-odds
added 1.4 / 15.7. q = 0.2: 5.6 / 14.8; calibrated 0.09 / 0.3 (the gate refuses nearly everything: a judge that misreads 20 % of fields
cannot reach 0.6 on four fields); two judges 0.81 / 6.0. At q = 0.1 the gate at 0.6 still lets harmful writes through with two
judges; the gate level is the pipeline's own policy knob (TAU_EXTRACT) and was not varied here.
More lenses do not remove an error that every lens of one judge shares; a second channel with a different origin does.
Calibration (test): stated 0.60, entirely right 0.64 (0.9⁴ = 0.66); uncalibrated states 0.81–0.92 and is right 0.60–0.65. A
straight-line (Platt) calibration was tried first and was under-confident (0.44 stated, 0.63 right); isotonic is used.

**Speed of the open typed-decision mechanism (e13; Qwen2.5-0.5B-Instruct fp16 on an RTX 5070; one 400-token state, 20 yes/no
fields).** One forward pass per field 504 ms per state (40 fields/s); all 20 as one batch with the state re-read 490 ms; state
prefilled once and its KV cache shared by the 20 field suffixes 54.5 ms (367 fields/s). The three give the same probabilities
(max difference 0.006, fp16). The open replica of the hosted service (read by an agent) does the same: prefill once, tile the
cache, one batched pass, softmax restricted to the option tokens, no training.

**Two real channels on one field (e14; the 320 labelled TEST sentences of e7).** Symbolic rule: coverage 1.0, accuracy 1.0 on the
templated sentences and 10/10 on hand-written sentences outside the templates (tests) — templates and lexicon have the same author,
so this is not an independent evaluation of the rule. Language model, eight lenses pooled: 0.591. Where both answer and agree
(59 % of sentences): 1.000; where they disagree the language model is right 0.000. Language model calibrated per sentence form on the
RULE's answers instead of the labels, held-out quantity pairs: 0.927 (sd 0.025), equal to calibration on true labels (e7b).

**The rule on text by other people (e15; QuaRTz, Tafjord et al. 2019, CC BY 4.0, 405 annotated paragraphs).** Given the paragraph
and the two annotated property phrases: answers 40 % (86 % when both phrases occur verbatim), right 0.901 where it answers;
majority class 0.709. After extending the comparative lexicon on the train split only: test split (81 paragraphs) coverage 0.43,
accuracy where answered 0.857. The same-author 1.0 of e14 does not transfer; 0.86 with abstention on the rest is the number to use.

**One currency for probing and working (e16, SIMULATION; 8 open nodes, one goal, cheap judge cost 1 with reliability r, cell
cost 20 exact, 600 graphs).** Spend per goal / wrong settlements per goal at admission level τ = 0.99, r = 0.8: existing `priority`
with cells only 53.6 / 0.000; (1 − p)/c order with cells only 35.7 / 0.000; judge first then cell 10.3 / 0.007; plan_value 10.6 /
0.015. At τ = 0.9 the cheap judge settles most nodes on one answer and wrong settlements rise to 0.05–0.15 per goal: cost alone is
gameable, τ is the knob. Negative: choosing the instrument per node by the settle-cost programme gives nothing over "judge first"
in this range, because the judge is always worth trying here; the programme skips a cheap instrument only when it cannot leave the
band or costs more than the cell (test). A first version scored probes by criticality × E[Δp]; that quantity is identically zero
(martingale) and was replaced before any number was produced.

**Record admission with a guarantee (test; e12's simulated judge, 4 fields, 10 % shared misreading → 0.64 of records entirely
right).** The product gate Π max(p, 1−p) ≥ 0.9 admits records that are right ~0.65 of the time (level broken in 20 of 20 splits).
`record_guarantee` at α = 0.1 admits nothing (the level cannot be certified), at α = 0.45 admits 30 %+ and holds the level in 20 of
20 splits (calibration 600 records, δ = 0.05). Two wrong versions were caught by the held-out test: scanning from the most certain
record admitted nothing; scanning until a threshold passed broke the level in 8 of 20 splits. Bonferroni over a fixed grid fixed it.

**Per-origin reliability (test; 300 questions, five origins with planted accuracies 0.9 / 0.85 / 0.8 / 0.55 / 0.5, no labels).**
Recovered within 0.07 of the planted values; fused signs with the estimated per-origin weights are > 3 points more accurate than
with one fixed reliability. Needs ≥ 3 independent origins per question to be identifiable; shared misreadings stay invisible.

**Mechanism signature (tests).** Electrostatic pull-in, Semenov thermal runaway and shallow-truss snap-through: one limit point,
order 2.00, β 0.50, γ 0.50, at (1/3, 4/27), (1, 1/e), (1 − 1/√3, ·). Symmetric column: odd, γ 1.00. Cusp, linear, saturating: no limit point.
Same thing, different names (e17): eleven named textbook phenomena from six fields, grouped by computed signature blind to the
names — the five folds (pull-in, thermal explosion, snap-through, harvested-population collapse, AFM jump-to-contact; five fields) in
one group, the two pitchforks (Euler buckling, Landau ferromagnetic ordering) in another, the rest without a limit point; every group
is one normal form.
Vector state (test): the scalar pull-in traced as a 1-vector gives the same numbers; a two-state shallow truss (vertical drop + an
asymmetric mode coupled through a spring) reads its snap-through fold at the known load with order 2.00 and γ = 0.50.

## Limits

- The label in e3/e8 is what a field did next, not what was worth doing.
- `margin_net`, `regime_posterior`, `claim_federation`, `throws`, `unlock_value` are verified on TEST data and simulation only.
- `regime_posterior` assumes at most two transitions along ONE variable; `mechanism_signature` handles one state variable.
- N_eff counts a single review that cites two independent origins as 2.
- e11: the ordering proof assumes independent nodes and work that stops at the first failure; dependence was measured up to a shared-cause probability of 0.3 only.
- e12: the judge is simulated. No real judge has been run through `typed_extraction`. e14 uses the real 0.5B model's answers and a real rule, on templated sentences.
- `polarity_rules`: English lexicon; one sentence; no coreference beyond "it"; abstains otherwise. Independent accuracy 0.86 (e15), not 1.0.
- `plan_value`: goals summed with a node under two goals counted twice; the order is the chain-segment rule, heuristic beyond chains.
