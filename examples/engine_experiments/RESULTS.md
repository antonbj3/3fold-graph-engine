# Engine additions: contract, what each module computes, and what was measured

Every number below is produced by a script in this directory (`python3 examples/engine_experiments/<script>.py`;
seeds fixed; result JSON next to the script). e1, e3, e3b, e3c, e6, e8, e8b download the public SNAP graphs
cit-HepTh / cit-HepPh to `$HUNT_DATA`; e19 also uses the SNAP cit-HepTh abstracts; e23 fetches hep-ex abstracts from the
arXiv API into `$HUNT_DATA`. e7, e13, e14, e20, e22 need a local Qwen2.5-Instruct (`$JUDGE_MODEL`, 0.5B / 1.5B / 3B; `JUDGE_TAG`
suffixes the output; `E7_DEVICE=cuda`); the 0.5B answers are shipped as `e7_probabilities.npy` (1.5B and 3B as `_1p5b`, `_3b`),
so e7b, e9, e10, e18 run without the model. e15 and e22 use QuaRTz (Tafjord et al. 2019, CC BY 4.0) from the Hugging Face hub.
Test graphs with planted truth are labelled TEST; `curated_two_graphs.json` is a hand-written fixture from public
textbook relations. No private graph is included or was used for any number in this file.

## Contract (what data has to look like)

A graph is a JSON object. All four record types share `sources`, the lineage that decides what counts as independent.

```
sources   [{"id": "Hall1951", "derives_from": []}, {"id": "Textbook", "derives_from": ["Hall1951", "Petch1953"]},
           {"id": "CDF-2012", "shares": {"CDF": 0.5}}]        three lineage states: independent / copy / partially shared error
concepts  [{"id": "grain_size", "aliases": ["d", "mean grain diameter"]}]   the alias table two graphs are joined through
claims    [{"id", "subject", "object", "sign": +1|-1, "validity": {"grain_size_nm": [25, 1e5]}, "evidence": [source ids], "cost"}]
edges     [{"id", "between": [var, var], "reports": [{"margin": 0.12, "sigma": 0.05, "sources": [ids], "validity": {...},
                                                      "attributes": {"age": 63, "site": "B"}}], "weight", "cost"}]
                                                     attributes = what is known about the report's setting; hidden_variable reads them
scales    {"grain_size_nm": "log"}                       optional; default linear
profile   EngineProfile as JSON (profile.py): the owner's assumptions, validated; shipped next to the graph
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
| `record_guarantee` | admit extracted records so that ≥ 1 − α of the admitted are entirely right, distribution-free | score = least certain field; threshold = largest of the unique score quantiles (≤ 20, taken from the calibration scores) whose Clopper–Pearson (1 − δ/grid size) bound is ≤ α (Learn-then-Test; valid because the labels are independent given the scores); assumes exchangeability only |
| `source_reliability` | how often each independent origin is right, from agreement, plus a few pinned truths to break the mirror symmetry | Dawid–Skene EM with a Beta prior over roots (copies collapsed first); feeds `Federation(reliability_by_root=…)` |
| `profile` | every assumption a graph's owner makes, in one validated object | ~30 fields grouped by the module that reads them; estimable ones (per-root reliability, p_transition) filled from data, requirements (z, p, τ, α) declared; `merge` of two graphs' profiles keeps per-root numbers and takes the stricter requirement, reporting every difference |
| `hidden_variable` | an overlapping-box disagreement read as a candidate UNDECLARED variable, not a contradiction to settle | data side: the attribute split with the largest χ² (or Gini) drop, permutation p-value, the boxes it would add; mechanism side (`from_mechanism`): the model parameters that can flip the sign of dy/dx inside the declared ranges, before any report |
| `precision_form` | one quadratic form for structure, margins and regimes; value of one observation or of a SET in bits | J from the Laplacian + lineage-weighted observation blocks + a local Gaussian image of a Bernoulli belief; rank-1 updates; set value ½ log det(I + H C Hᵀ/σ²), submodular, greedy |
| `numeric_rules` | numbers with uncertainty out of text: v ± s, asymmetric, stat ⊕ syst, CI, ranges, powers of ten, unit prefixes | deterministic; abstains when no quantity phrase is found; feeds `margin_net` (value = margin, σ) |
| `provenance_rules` | who a report derives from, out of the text itself | copy / derived / shared / cites → `derives_from` or `shares`; abstains when a marker names no source |
| `regime_markov` | the sign along x as a telegraph process: any number of transitions in one code path | forward–backward on the fixed partition, exact; majority box claims exact by block enumeration; P(N = k) from the same pass |
| `next_actions` | one list of what to do next, from every channel, in the contract's form; a runner that routes the outcome back | bits per cost across regimes, margins and the precision form; priority / χ² / EVPI channels listed apart, never mixed in; `apply` returns the fold-ledger row with predicted and realized bits |
| `claim_types` | the checker layer: dimensions, certificates, link typecheck | unit grammar → dimension vector; a claim carries a certificate (deductive from a model over its box, statistical from a fitted guarantee, or none) and readers take r from it; a two-step link must resolve one concept, intersect boxes, agree on scale and dimension |
| `decision_cert` | trust relative to a DECISION, not global (from the Hollow project's decision certificates): p_flip of a predicate over the beliefs, and which measurement can flip it | sign-on-box (exact posterior mass), margin (Φ), chain (declared independence); `flip_attribution` ranks candidates by P(flip)/cost; listed in `next_actions` as a separate currency |
| `closed_loop` | the engine choosing probes against a world with known sign structure, scored against the truth | world of ≤ 1-transition sign functions with sourced, copied, unreliable claims; policies engine / copies / random / oracle; wrong measure and believed wrong measure |
| `polarity_rules` | the sign a sentence asserts between two quantities, symbolically | one direction word per quantity per clause, negation flips, last clause wins, composition by product; abstains outside its lexicon (English only) |
| `representation_probe` | a second reader: a linear direction in a frozen model's mid layers, trained on the rule's labels, with its own lineage | reads text that contradicts the model's prior where the token output does not; inherits the genre of its training sentences (e22) |
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

**A throw is a set, not a pair — mostly a degree effect (e19; 19 674 papers in cit-HepTh/HepPh at two cutoffs).** For each new paper:
core = median resistance distance between its references, throw = 90th percentile; impact = citation percentile within its month. First
reading: core tightness carries the impact (Spearman −0.14 to −0.27 in all four cells), a far throw is negative on its own but positive
given the core (partial +0.02 to +0.11), and the tight-core-far-throw quartile has the highest mean impact in 4 of 4 cells (0.59–0.66 vs
0.47–0.51) — Uzzi et al. 2013 in resistance geometry. Control, after the question "can this be biased?": resistance distance between two
nodes is ≈ 1/deg_i + 1/deg_j, so "tight core" is mostly "the references are hubs" (Spearman −0.80 to −0.90 between the core measure and
the references' mean log in-degree). Given hubness and reference count, core→impact falls to −0.02 to −0.07 and throw→impact given core to
+0.005 to +0.05. What the geometry mostly measured is that papers citing popular papers get cited more. Biases that no control on this
data removes: references outside hep-th/hep-ph are not in the graph (a far throw out of the field is invisible), only published papers
exist (rejected far-throw-no-core papers are missing), and used as a policy the pattern rewards itself. `throws` scores pairs; a set
score is not built, and on this evidence it should not be built from core tightness.

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
(max difference 0.009, fp16; an exact-arithmetic check on a random-weight model gives 0.0, so the difference is fp16 noise and does not grow with the spread of suffix lengths). The open replica of the hosted service (read by an agent) does the same: prefill once, tile the
cache, one batched pass, softmax restricted to the option tokens, no training.

**Two real channels on one field (e14; the 320 labelled TEST sentences of e7).** Symbolic rule: coverage 1.0, accuracy 1.0 on the
templated sentences and 9/9 on hand-written sentences outside the templates (tests) — templates and lexicon have the same author,
so this is not an independent evaluation of the rule. Language model, eight lenses pooled: 0.591. Where both answer and agree
(59 % of sentences): 1.000; where they disagree the language model is right 0.000. Language model calibrated per sentence form on the
RULE's answers instead of the labels, held-out quantity pairs: 0.927 (sd 0.025), equal to calibration on true labels (e7b).

**The rule on text by other people (e15; QuaRTz, Tafjord et al. 2019, CC BY 4.0, 405 annotated paragraphs).** Given the paragraph
and the two annotated property phrases: answers 40 % (86 % when both phrases occur verbatim), right 0.901 where it answers;
majority class 0.709 over all 405. After extending the comparative lexicon: test split (81 paragraphs) coverage 0.42, 29 of 34 answered
right = 0.853, 95 % Clopper–Pearson [0.69, 0.95], majority class among the answered 0.676. Provenance caveat: the lexicon extension, the
test-split print and the module docstring landed in one commit, so "train split only" cannot be verified from the history; treat the
test number as a same-author number on a small sample. The same-author 1.0 of e14 does not transfer; ≈ 0.85 on n = 34 with abstention
on the rest is what the rule has shown on other people's text. After the review's abstention fixes (cross-sentence composition, third
quantities, parentheticals inside the negation window, double negation) the split is unchanged within one item.

**One currency for probing and working (e16, SIMULATION; 8 open nodes, one goal, cheap judge cost 1 with reliability r, cell
cost 20 exact, 600 graphs per cell, common random numbers).** Spend per goal / wrong settlements per node worked, τ = 0.99: existing
`priority` with cells only 50.8 / 0; (1 − p)/c order with cells only 35.9 / 0; judge first then cell at r = 0.8: 10.7 / 0.006; plan_value
10.8 / 0.007. Review found the first version's comparison to be an ordering artefact (work stops at the first failure, so different orders
work different node sets, and "wrong per goal" is not normalized by nodes worked). Redone with fixed-order arms: at r ≥ 0.7 the two
instrument rules are the same policy (the settle-cost programme also picks the judge first everywhere), so the instrument choice is
neutral there. At low judge reliability the programme is what matters: τ = 0.90, r = 0.55: judge first 142 / 0.086, plan_value 35.7 /
0.011 (8× fewer wrong settlements per node worked, 4× cheaper); τ = 0.99, r = 0.55: 401 / 0.009 against 36.9 / 0.000. Cost alone is
still gameable: at τ = 0.9 the cheap judge settles nodes on one answer and wrong settlements rise; τ is the knob. The settle-cost
programme itself had a fixed horizon of 6 steps with an infinite terminal value; it now doubles the horizon until the value converges
(from below). A first version scored probes by criticality × E[Δp]; that quantity is identically zero (martingale) and was replaced
before any number was produced.

**Record admission with a guarantee (test; e12's simulated judge, 4 fields, 10 % shared misreading → 0.64 of records entirely
right).** The product gate Π max(p, 1−p) ≥ 0.9 admits records that are right ~0.65 of the time (level broken in 20 of 20 splits).
`record_guarantee` at α = 0.1 admits nothing (the level cannot be certified). At α = 0.45 it admits 99.8 % — every score bin of that
population has error 0.35–0.39, so no threshold selects and "holds the level" there only says the base rate is below α (a first draft
wrote "30 %+", which understated the coverage by 3× and made the demonstration look like a gate; found in review). On a heterogeneous
population (half a strong judge with 2 % shared misreading, half a weak one with 12 %; base error 0.40, best bin 0.07) the threshold
is interior: α = 0.20 read off the calibration data admits 0.52 (0.45–0.62 over 20 splits) with admitted error 0.117 mean, 0.172 max,
level held 20 of 20 (calibration 600 records, δ = 0.05). Two wrong versions were caught by the held-out test: scanning from the most
certain record admitted nothing; scanning until a threshold passed broke the level in 8 of 20 splits. Bonferroni over the quantile
grid (the unique quantiles of the calibration scores; valid because the labels are independent of the grid given the scores) fixed it.

**Per-origin reliability (test; 300 questions, five origins with planted accuracies 0.9 / 0.85 / 0.8 / 0.55 / 0.5, no labels).**
Recovered within 0.07 of the planted values; fused signs with the estimated per-origin weights are 1.3–12 points (mean 7 over 10 seeds) more accurate than
with one fixed reliability. Needs ≥ 3 independent origins per question (two identify only their agreement rate); shared misreadings stay
invisible. Not identifiable from agreement alone: the likelihood is symmetric under flipping all truths and all reliabilities, so a
majority of systematically wrong origins comes out mirrored (review planted 0.9 / 0.2 / 0.2 / 0.15 / 0.15 and got 0.07 for the good one,
its vote negated). Without an anchor the estimate assumes origins are mostly better than a coin; `known` = a few pinned truths (20 of
400 suffice in the test) breaks the symmetry from data.

**Model size (e7 rerun with Qwen2.5-1.5B and 3B on GPU; same 320 sentences, same eight lenses).**

| | 0.5B | 1.5B | 3B |
|---|---|---|---|
| right per lens | 0.47–0.59 | 0.75–0.80 | 0.83–0.88 |
| pooled, eight lenses | 0.591 | 0.797 | 0.869 |
| right when the text agrees with the physics / contradicts it | 0.57 / 0.50 | 0.92 / 0.63 | 0.995 / 0.71 |
| error correlation, same / swapped option order | +0.61 / −0.08 | +0.67 / +0.58 | +0.75 / +0.74 |
| N_eff of the eight lenses | 3.2 | 1.5 | 1.3 |
| composition templates ("reducing X lowers Y") | 0.21–0.35 | 0.43–0.59 | 0.77–0.83 |
| calibrated per form on the rule's labels, held-out pairs (20 splits) | **0.927** | 0.853 | 0.855 |

Bigger models read better and believe their prior more: the gap between text that agrees with the physics and text that contradicts it
grows from 7 to 29 points, and stays at 29 from 1.5B to 3B. The lenses stop being different lenses (N_eff 3.2 → 1.3; the swapped
option order no longer gives an opposite channel). The 0.5B's errors were structural (option position, sentence form) and per-form
calibration on the rule's labels removes them; the larger models' errors are knowledge-driven and calibration cannot reach them,
so after calibration the smallest model is the most accurate. The sentence forms here are the rule's own domain (rule 320/320), so
this measures the reading channel, not the rule. Consequence: the reader that belongs next to the symbolic channel is a small model
whose position and form biases are calibrated away, not a larger model; a larger model contributes a prior, which the federation
must keep on a separate lineage from the text.

**Reading the representation before the collapse to a token (e20; same 320 sentences, lens 0, hidden state of the last prompt
token at every layer, linear probe trained on the RULE's labels over 10 quantity pairs and tested on the other 10, 20 splits; the
layer is chosen on the training pairs only).**

| | 0.5B | 1.5B | 3B |
|---|---|---|---|
| token output: all / text contradicts physics | 0.52 / 0.48 | 0.78 / 0.64 | 0.85 / 0.71 |
| linear probe, layer chosen on training pairs | 0.86 / 0.87 | 0.89 / 0.83 | **0.92 / 0.87** |
| probe on the last layer, contradicting text | 0.86 | 0.80 | 0.71 |

The reading lives in the middle of the network (best layers 18 of 25, 21 of 29, 24 of 37) and the prior is applied in the last
layers: in the 3B the probe on contradicting text falls from 0.92 at layer 21 to 0.71 at the output — the token output's number — and
in the 0.5B it does not fall at all. So "a larger model reads worse" (model-size table above) is a statement about the output head.
The representation of the larger model reads better than the smaller one's and without the model's belief, when it is read before
the collapse. The probe is trained on the symbolic rule's labels, no human labels; it is a second reader whose errors are not the
rule's (the rule is silent outside its forms, the probe is not) and not the output head's. Prior work: Burns et al. 2022 (probe beats
output by 4 points), Li et al. 2023 (steering along a probe direction). Built as `representation_probe` (a lens with lineage tag `probe:<model>:<layer>`) and tested on other people's text (e22, QuaRTz
test split, 81 paragraphs, majority 0.716): the probe trained on OUR templates with the rule's labels does not transfer (0.57 / 0.88 /
0.57 for 0.5B / 1.5B / 3B — it inherits the rule's domain of single-sentence direction words); trained on QuaRTz's own 283 training
paragraphs it reads 0.80 / 0.91 / 0.90, and 0.85 / 0.94 / 0.89 on the 47 items where the rule abstains, so it covers what the rule
cannot. And the token output on QuaRTz is NOT worse than the probe (0.83 / 0.89 / 0.93): QuaRTz paragraphs mostly agree with physics,
so the prior helps there. The e20 gap is therefore scoped to text that contradicts the model's prior — exactly the text a research
graph is for (a surprising measurement, a contradiction between sources) — and on ordinary text the output head is as good. n = 81:
one item is 1.2 points.

**The reader trained on real text with the rule's labels (e25).** The rule labels real sentences for free: 726 hep-ex abstract sentences
(2.6 % of all sentences; a heuristic (x, y) span finder) and 108 QuaRTz-train sentences (the rule agrees with the human paragraph
label on 0.917 of them). Probes trained on those, tested on QuaRTz test (n = 81): hep-ex only 0.77 / 0.82 / 0.88 (0.5B / 1.5B / 3B);
QuaRTz-train rule-labelled only (68 sentences) 0.75 / 0.84 / 0.91; both 0.84 / 0.85 / 0.88 — against 0.57 / 0.88 / 0.57 from the
templates and 0.80 / 0.91 / 0.90 from 283 human-labelled paragraphs. Hand check of 20 rule-labelled hep-ex sentences: the sign is
right in 15, the full (x, y, sign) triple in 4 — the span finder, not the rule, is the weak part, and the probe transfers anyway.
Verdict: the training GENRE is what carries, not label quantity or purity; 660 noisy real sentences beat 320 perfect templates by
20+ points on other people's paragraphs, and free labels come within 3–4 points of human ones. Cost: the template set falls
(0.86 / 0.89 / 0.92 → 0.63 / 0.76 / 0.79), and on QuaRTz no probe beats the token output (the paragraphs agree with the prior).

**Negative: the prior cannot be projected out of the token channel (e29).** The difference-of-means direction between text that agrees
with physics and text that contradicts it is real in the last state (held-out AUC 0.73 at 3B, 0.50 at 0.5B where there is no gap) and
orthogonal to the reading direction (|cos| < 0.03), but removing it — or steering along it from one class mean to the other — moves the
A/B output by 0.03 and never raises accuracy on contradicting text (0.701 → 0.660…0.694 at 3B). The prior is not one linear direction
in the final state; it is applied by the last layers in a way one projection does not undo. The mid-layer probe stays the only reader of
contradicting text (0.87 at e20's layer vs 0.71 for a probe on the last layer).

**The loop closed against a world that answers (e21, `closed_loop.py`; 40 worlds × 12 pairs; per pair the true sign along x changes at
most once, 5 % of pairs twice; 8 sources with reliabilities 0.6–0.95, 35 % of them copies of another source; instruments: judge cost 1
r = 0.8, exact cell cost 4 r = 0.99; budget 40; score = measure of the domain wrongly signed against the truth, summed over pairs).**
Wrong measure at budget 0 → 40: random probes 2.27 → 2.00; engine (expected potential drop per cost, lineage-weighted claims)
2.27 → 1.24; paired difference −0.76 ± 0.09, engine better in 37 of 40 worlds. Copies counted as independent: 2.48 → 1.29; paired
against the engine −0.04 ± 0.05 (22 of 40) — on accuracy the lineage weighting gives nothing here, because a copy repeats a report that
is mostly right and the extra confidence lands on the right side; where it costs is calibration: after the budget the engine believes it has 0.97 wrongly signed and has 1.20 (gap 0.23 ± 0.10, 12 worlds); with copies counted as independent it believes 0.76 and has 1.24 (gap 0.47 ± 0.08). Both are over-confident (the claim reliability inside the posterior is a fixed 0.75 while probes enter at their own reliability); counting copies doubles the over-confidence. Negative: the
collision family did not flag the two-transition pairs (engine 0 of 25, copies 2 of 25, 1–2 false flags) at this budget — the
two-transition posterior needs probes on both sides of both transitions, and the value rule never buys them because the
single-transition reading explains the claims. The oracle policy is implemented but not run (25 × 12 × 2 posterior copies per step).
This is the first measurement in this package of the engine choosing experiments end to end; the world is functions with the sign
structure of e17's mechanisms, not a simulator.

Three additions to the loop, same 40 worlds, paired against `engine` (1.24 at budget 40):
- **guard** (a share of the budget bought against the model error with `model_check_probe`): the first collision flags in this
  package — 3 of 25 two-transition pairs at share 0.2 or 0.3 with 0–1 false flags (0 of 25 before), accuracy cost +0.04 ± 0.04.
  Share 0.1 is worse on both counts. Kept, default 0.2.
- **online reliability** (Dawid–Skene per root with the probes as pinned truths): HARMFUL, +0.085 ± 0.036 wrong and the calibration gap
  grows 0.21 → 0.56. It refuted its own premise: the estimated root reliabilities are 0.80–0.93, above the fixed 0.75, so honest
  per-source weights make the engine more confident while the truth does not move. The over-confidence is not in r — it is in the
  pointwise reading of a box claim (a source that reports the majority sign of a wide box is read as r-accurate at every point). The
  estimator itself works (a planted 0.5 source is ranked under a planted 0.98 one after 20 probes, test).
- **replay** (empirical Bayes on the loop's own history: p_two, reliability and p_flip re-estimated from the posterior family masses
  of earlier worlds and used as the next world's priors): −0.085 ± 0.037 (22 of 40), growing with history — first 10 worlds −0.08,
  last 10 −0.26 — and the calibration gap 0.28 → 0.12 while the fixed-prior engine's stays at 0.27–0.29. Learned after 40 worlds:
  p_two 0.052 (true 0.05), reliability 0.82 (true 0.88), p_flip 0.49 (true 0.63; a transition no probe brackets keeps its family at
  prior mass, so the fixed point sits low — pinned in a test, not asserted away). This is the recursion that works: not on which
  episodes to replay (e3, zero), not on the residual alone (e3c, +0.01–0.02), but on the loop's own priors from its own outcomes.
- all three together: collision 4 of 25 with 0 false flags, accuracy +0.03 ± 0.05; the reliability part eats replay's calibration gain.
- **reading a box claim as a majority report** (`RegimePosterior(claim_model="majority")`: the source reported the majority sign of its
  box, likelihood r·S(k(f−½)) instead of r-accurate at every point): the calibration gap halves, 0.23 ± 0.10 → 0.12 ± 0.08, and the
  wrong measure is not worse (1.20 → 1.17; 12 worlds). This is the fix the reliability experiment pointed at. Kept as an option,
  exposed in the profile; the default stays pointwise so the earlier e4 numbers remain reproducible.
- **pairs of model-check probes + replay + majority** (`model_check_pair`: the two cells whose joint answers drop the family entropy
  most, exact over four outcomes): pairs ALONE regress (1 of 25, wrong +0.135 — a pair costs twice as much and the two answers usually
  confirm one transition); with replay 6 of 25; with replay and the majority reading **7 of 25 with 2 false flags, wrong measure 1.08
  (−0.164 ± 0.053 against the engine, 27 of 40 wins), calibration gap −0.08** (slightly under-confident). This is the best configuration
  measured: collisions found only when the prior p_two is learned from history and the box claims are read as majority reports, so
  the single-transition family is no longer forced to fit. e28 explains the rest of the hole: with 12 probes on one pair both readers
  find the second transition in 31 of 40; the loop gives each pair ~3. The remaining 18 of 25 are an allocation problem.
- **burst allocation (e21b): negative.** Committing a run of 6 (or 4) model-check probes to the pair the guard picks, each placed
  after the previous answer: 5 of 25 with 0 (or 3) false flags, wrong 1.08 / 1.11 — not better than pairs (7 of 25). The guard's
  20 % of a 40-unit budget is 8 probes: one burst per world, on one pair. Allocation is not fixable inside a budget of 2–3 probes
  per pair; a second transition costs ~12 probes and the budget has to be raised for it or a cheaper instrument (a sweep from a
  computation cell: many probes per run) has to exist. Recorded as the limit, not as a rule to tune.
- **E-optimal / T-optimal allocation (e21c): negative, with the reason.** A max-min rule (raise the smallest pairwise discrimination
  between mass-carrying hypotheses; the identifiability-steered excitation of the Hollow project, Atkinson–Fedorov's model
  discrimination) is identically zero on this fixed threshold family until nearly every cell is probed: two hypotheses that agree on
  every probed cell have discrimination 0, and one probe lifts only the pairs straddling its own cell. The fallback (mean lift of the
  tied weakest set) spends probes on "threshold in cell i vs i + 1" — hairline distinctions of no value. Measured: +0.10 ± 0.05 wrong and
  4 of 25 collisions (baseline 7), the mixed rule +0.17. The second transition is not unidentifiable in the max-min sense; it is
  LOW-PRIOR, and what buys it is the pair purchase with a learned p_two and the majority reading. Two allocation rules refuted
  (burst, E-optimal); the diagnosis stands as a budget statement.

**One quadratic form (e24, `precision_form`).** Exact to machine precision: effective resistance = (e_i−e_j)ᵀJ⁺(e_i−e_j) (1e-16);
the GLS estimate and variance of `margin_net` = posterior mean and variance of the observation block (0.0 / 4e-18, including a
singular Σ from declared copies); a report is a rank-1 update (Sherman–Morrison, 2e-15; an update in the Laplacian's null space is
detected and recomputed exactly instead); the value of an observation ½ log₂(1 + hᵀCh/σ²) equals the realized drop of the Gaussian
entropy (2e-14) and the value of a SET is the sum along any order (chain rule, 9e-16). The Bernoulli block is a LOCAL Gaussian
image: it ranks candidate probes as `regime_posterior` does (Spearman 0.9999) but its bits are off by up to 43 % (median 15 %) against
the exact two-outcome drop — order yes, bits no. Set value is monotone and submodular (200/200 random checks), greedy found the
optimum in 100/100 small instances. Set composition on a two-cluster graph: the best triple is three far pairs with low mutual
coherence, not "a tight core plus one far element" (the throws.py expectation, now contradicted twice: e19 and here) — the rule trades
resistance for independence: more bits than the top-3-by-resistance in 6/6 instances (2.2–2.5 vs 2.1–2.3; random 1.6–1.9). One bits-per-
cost list ranks structure, margin and regime candidates together; the three single-currency top-1s are three different candidates and
the joint top-1 is a fourth that none of them nominates, because cost and probe noise are invisible to resistance alone.
Five readings of one matrix (added the same evening): hole field = diag(C) (error 0.0 against the exact pinv), edge leverage
w_e R_e with Σ = n − 1 (7e-15), a set throw as the DPP with kernel Q C_S Q/σ² (marginals equal `throws` to 8e-13 when the quality
absorbs the row normalisation — plain `draw_set_dpp(Z)` is a different kernel of the same family), a confirmed link as the rank-1
update with realized value ½ log₂(1 + wR_ij) (8e-15), and the spectral sparsifier from exact leverages (resistances kept within the
same band as the sketched version, mean 0.12 vs 0.14). Not the same object: the entropy drop and the Kirchhoff (trace) drop rank the
same 120 candidate links at Spearman 0.94 with different top-1s — a log-det ranking of links is not a trace ranking.

**Hidden variables (test).** Sixteen reports with a planted effect modifier (sign flips above age 50) among two decoys: the split on
age at 45–55 is found with permutation p < 0.02 and the decoys at p > 0.05; with no planted effect every attribute is at chance; an
attribute missing on most reports cannot score. Mechanism side: for harvested yield Y = hK(1 − h/r) against effort h ∈ [0.2, 0.8],
the growth rate r (flip at h = r/2) is nominated and the carrying capacity K (scales, never flips) scores 0.

**Conf with precision on a real experimental corpus (e23; 7 201 hep-ex abstracts from the arXiv API, cached outside the repo).**
`numeric_rules` extracts a numeric claim with uncertainty from 11.7 % of abstracts (bare numbers without uncertainty are not
extracted). Hand check of 30 random abstracts, all read: 2 right, 1 wrong (junk quantity phrase on a range), 3 abstained where a
number was there (a paper-private LaTeX macro; twice a single Greek-letter quantity the phrase rule rejects), 24 correct abstentions.
Then `margin_net` on the quantities many papers report, each paper its own source, and again with the collaboration as the root:

| quantity | n | per paper m̂ ± s (N_eff) | per collaboration m̂ ± s (N_eff) | χ² p, per paper | PDG |
|---|---|---|---|---|---|
| top quark mass, GeV | 93 | 172.73 ± 0.11 (93) | 172.75 ± 0.64 (10) | 0.99 | 172.60 ± 0.27 |
| W mass, GeV | 35 | 80.3750 ± 0.0032 (35) | 80.364 ± 0.005 (14) | 9e-6 | 80.3625 ± 0.0077 |
| Higgs mass, GeV | 9 | 125.23 ± 0.07 (9) | 125.35 ± 0.28 (2) | 0.55 | 125.13 ± 0.11 |

Per paper, 37 CDF and 17 D0 papers count as independent and s is 3–6× too small; with the collaboration as root the estimates move onto
the PDG values (W 1.5σ → 0.1σ). The χ² fires for W (Q = 81 on 34 dof: the LEP/Tevatron/LHC spread is wider than the quoted σ) and
does NOT fire for the top mass (Q = 64 on 92 dof: quoted total σ are larger than the scatter, and abstracts re-quote combinations),
so the test does not see the known Tevatron/LHC offset (≤ 2011: 172.86 ± 0.30; ≥ 2012: 172.71 ± 0.12; PDG's Tevatron average
174.30 ± 0.65) — the offset is inside the individual σ. Limit found by the run: with the collaboration as root, margin_net's copy check
(b) fires on every quantity, and it is right to — successive measurements by one collaboration are not copies of one report, they
are different measurements sharing part of their systematics. The lineage model has two states (independent / copy) and this
corpus needs a third: partially shared error, e = D(√(1−ρ) ε_own + √ρ ε_root). Built (`shares: {group: ρ}` on a source) and rerun
with ρ = 0.5 within a collaboration: top mass 172.54 ± 0.12 (N_eff 73, p 0.09, closer to PDG than either other mode), Higgs
125.27 ± 0.09 (N_eff 5), W unchanged in value with p smaller still (the spread is real). The copy check no longer fires, as
intended. ρ = 0.5 is a declared assumption, not a measurement; what a collaboration actually shares between two of its
measurements is in the papers' systematic tables, not in abstracts, and the honest use is to let the graph's owner declare ρ per
group and see the three readings side by side.

**Provenance from text on a medical corpus (e26; 5 814 PubMed abstracts on blood-pressure targets, NCBI E-utilities, cached outside
the repo).** The owner's point: in medicine, authority is many copies of a few trials, and nothing declares it. `provenance_rules`
finds a provenance statement in 16.9 % of abstracts and a NAMED lineage root in 11.2 %; in 36.7 % a marker fires ("based on",
"following the guideline") and names nothing — the literature claims provenance three times more often than it names it. Roots:
SPRINT 151, ACCORD 15, STEP 15, ACC/AHA guideline 15 … (586 distinct). Hand check, all read: 21 of 25 extractions right (4 wrong:
an acronym that was the paper's own arm, a truncated name, a risk factor read as a source), 12 of 15 empty outputs correct
abstentions, 3 missed. On the hazard ratio for cardiovascular events under intensive vs standard control (73 abstracts): independent
0.718 ± 0.006 (N_eff 73); copies of the declared root 0.720 ± 0.007 (N_eff 67); shared ρ = 0.5: 0.715 ± 0.006 (71.7). χ² rejects
"one number" in every mode (endpoints, populations and follow-ups differ, as they should). The lineage correction is capped by the
11 % declaration rate, not by the model: where the text does not name its root the engine cannot collapse copies — the reading
channel for "according to the guideline" without a name is the next hole. Found on the way: `numeric_rules` reads only 11 % of the
12 135 "95 % CI" mentions in this corpus (the value must sit next to the interval); a surface normalisation in e26 lifts it to 65 %
— ported into `numeric_rules` (grammar-level: separator and unit between value and interval; the label before the value as the
quantity; ratio quantities HR/OR/RR get σ on the LOG scale with `log_scale: True`): 75.4 % of the 12 135 mentions read; hand check of
20 new extractions 17 right, 3 wrong (one of them the abstract's own misprint, flagged `value_outside_ci`).

**The regime posterior as a Markov chain (e28, `regime_markov`).** Sign along x as a telegraph process with switching rate λ, discretised
on the same fixed partition; forward–backward exact, the switch count carried in the message so P(N = k) comes from the same pass;
majority box claims read exactly by enumerating switch patterns inside each block of overlapping boxes (prior mass dropped by the cap:
≤ 0.012). Against `RegimePosterior(claim_model="majority")` on 200 worlds with the same claims and probes: expected error differs by 0.015
on average (max 0.12) — two models, one family. Two-transition pairs, 12 probes placed by each reader's own value rule, 40 worlds: the
chain reads P(N ≥ 2) > 0.5 in 31 of 40 and the enumeration's `collision()` also 31 of 40. So the e21 hole (0 of 25 flagged) was not
the reader and not the single-pair placement: with 12 probes on the pair both find it. It is the ALLOCATION across pairs — in e21 each
pair got ~3 probes and the value rule spends them where the sign potential falls, which is never on a pair the one-transition reading
already explains. Timing: a plain read 3× faster than the enumeration (0.6 vs 2.0 ms), probe selection slower (37 vs 16 ms) because
the chain re-solves per candidate. Kept as the reader for N ≥ 2; the enumeration stays for probe selection.

**Throws from the continuum (e27, e27b; cit-HepTh, T = 95, horizon 24, 600 sets of three; label = a new paper co-cites a pair of the
set that was not co-cited before).** Share of sets with a future link: random 0.7 %; far pairs by softmax 0 %; the MIDPOINT of two
nodes in resistance space decoded to its three nearest nodes 6.7 % (near pairs 7.2 %, far pairs 1.0 %); dither on the neighbourhood's
scale 1.7 % (dither on the distance from x*, the first default, 0.8 % — it scattered the decode; fixed); a CHAIN along the segment
between a far pair, decoded at t = 0.25 / 0.5 / 0.75, 2.0 % with within-set resistance 1.2 — the only rule that finds long links
above random (3×); a DPP set on a 150-node subsample 0.5 % (e31: at k = 64 the sketched kernel is 98.7 % below the noise floor — the draw was from noise); an OED-style point (largest hole pushed toward a far node) 0.8 %.
Reading: decoding a point of the continuum to nodes is a 10× candidate generator for co-citation, but the label rewards locality
(as e3b and e19 said), so near midpoints win; the chain is what makes a long throw pay at all. Throws are late-decoded points now;
what x* should be is open — the design point that maximizes variance found nothing here.
Would-be leverage (e27c): for a candidate link that does not exist, the probability it would sit in a uniform spanning tree of the
graph with it added is wR_ij/(1 + wR_ij) — exact (Sherman–Morrison; checked to 1e-10 against a fresh pinv). Decile curve over 20 000
random non-co-cited pairs: future co-citation is MONOTONE DECREASING in would-be leverage (bottom decile 1.0 %, top five deciles 1 of
10 000; Fisher p = 1e-9). A candidate that would be a bridge is anti-predictive; the mid band is empty, not a sweet spot. Third
time the same answer from three different scores (raw resistance, midpoint decode, would-be leverage): on citation data the label
rewards what is already almost connected. The quantity stays in `throws` as an exact bridge measure; it is not a link predictor.

**The checker layer (e30, `claim_types`; seed: proof checking blocks mistakes before they enter).** Dimensions over the raw numeric
extractions of both corpora: hep-ex 1 761 records, 6.9 % rejected (unknown unit 52, mass without unit 29, dimension mismatch 18);
PubMed 14 467 records, 10.4 % rejected (duration without unit 530, dimension mismatch 355, unknown unit 338, a ratio with a unit 143,
blood pressure without unit 102). Hand check against the sentences: 18 of 20 rejections right, 19 of 20 acceptances right; the
rejections mostly expose extractor defects ("mm Hg" split into metres, "years" refused, fb⁻¹ losing its exponent, "vs"/"LHC" read as
units). Certificates: a deductive certificate for Y = hK(1 − h/r) holds sign + on h ∈ [0.2, 0.4] over the declared parameter box and
fails on [0.2, 0.8] with a counterexample at h ≥ r/2 and the verified sub-interval; a reader takes r = 1 − 1e-6 from a deductive
certificate inside its box, 1 − α from a statistical one, the profile's default from none. Link typecheck rejects a two-step inference
whose shared variable is log on one side and linear on the other unless both declare it. Caveat: the modal acceptance is "no type
claim was made" — a dimension check cannot see a lost per-cent sign.

**Decision-relative certificates (test; `decision_cert`, lifted as a concept from the Hollow project).** A node with wide uncertainty
can still certify a decision if the decision does not flip anywhere in the credible set: p_flip is exact on the regime partition
(equals brute-force enumeration), Φ-exact for margins (Monte-Carlo agreement 3e-3), a declared-independence product for chains. The
attribution of the next measurement follows the decision, not the entropy: with two pairs where only one carries the decision, the
bits ranking puts the empty high-entropy pair first and the flip ranking puts the decision's pair first. Found on the way: a probe
OUTSIDE the decision's box can flip a sign decision (a far − answer lifts exactly the hypotheses that fail on the box) — a first test
assumed the opposite and was wrong.

**Coupled criticality and the noise floor (e31; two more readings of the precision form, concepts from the Hollow project).**
Criticality = the smallest non-gauge eigenvalue of J with its direction. Two healthy parts (σ_min 1.05 each) glued through one shared,
unmeasured concept: joint σ_min 0.050, a 21× collapse, and the weak mode loads 0.5 / 0.5 on the two interiors with the shared node at
amplitude exactly 0. Two things the seed did NOT transfer as stated: gluing precision forms is information-monotone, so a joint
weakness is always a cross-part mode, never inside one part (proved and tested) — the seed's dynamical instability has no analogue;
and "one measurement on the shared variable repairs it" is false: an absolute measurement is orthogonal to the mode and spends the
gauge (σ_min 0.050 → 0.017), while the CONTRAST across the seam raises it 3× (0.148). With asymmetric leans gluing can even improve
σ_min (1.49×); criticality needs the cross-part bottleneck. Noise floor (Marchenko–Pastur, upper edge σ²(1 + √(n/k))²): on a pure-noise
k-column Gram every eigenvalue is inside the bulk and the edge is met within 1.4–3.3 %; a planted spike is the one eigenvalue left
unflagged (overlap 0.995). Applied to the DPP kernel built from a resistance sketch of a 300-node graph: k = 16 → 99.7 % of the
eigenvalues below the floor, k = 64 → 98.7 %, k = 256 → 96 %; the exact kernel's entire spectrum (top 3.4) lies under the k = 256 floor
(4.3). So a determinantal throw drawn from a sketched kernel is drawn from noise at these widths — which is what e27's DPP arm
measured (0.5 %, below random). A σ_min below the floor is not a finding; a DPP on a sketch needs the exact kernel or k ≫ n.

**Label-free calibration (e18; same 320 sentences).** Raw pooled 0.591; subtracting each lens's batch-mean log-odds (Batch Calibration) 0.572;
per sentence form 0.603; per-form Platt on the rule's answers 0.927; on true labels 0.927. The option prior is not the fault; the
per-form sign inversion is, and only labels (the rule's are enough) fix it.

Sources behind the modules, with their numbers next to ours: LITERATURE.md.

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
