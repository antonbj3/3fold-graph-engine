"""Read-only graph-engine composition; biological identification requires explicit gates.

No endpoint measurements are passed to the predictive callable. Federation's
N_eff is structural information, not proof of measured error decorrelation.

This module contains no mathematics of its own: it composes six engine modules that nothing
else called together, and adds the coverage audit that explains a silent empty result.

What each imported piece contributes, so a reader need not open all six:
  hidden_variable.from_mechanism  mechanism side, before any report: sweep each parameter over
                                  its published range and measure the share of (x, theta) where
                                  the SIGN of dy/dx differs from the midpoint's, plus the
                                  threshold where the flip first appears. Zero flips is a result:
                                  the anchor's validity box suffices for the sign question.
  hidden_variable.candidates      data side: reports that disagree while their validity boxes
                                  OVERLAP are read as an undeclared axis, not a contradiction to
                                  settle. Ranked by chi-squared drop with a permutation p-value.
                                  It returns [] silently below its thresholds -- see
                                  coverage_audit(), which is why that function exists.
  claim_federation.Federation     traces every source to its root sources, so four papers citing
                                  one original support a claim as ONE, not four.
  margin_net.MarginNet            the margin, and the alarm when it goes negative.
  decision_cert.certify           the decision as a certificate rather than a number;
                                  flip_attribution names the quantity that would reverse it.

An anchor's DISPERSION is the fingerprint of its hidden variables. Matching only the mean is
POPULATION_FITTED, not resolved; that distinction is the point of running this at all.
"""
from __future__ import annotations

import dataclasses
import math
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm

from .hidden_variable import from_mechanism, candidates
from .claim_federation import Federation
from .margin_net import MarginNet
from .decision_cert import Decision, certify, flip_attribution
from .next_actions import Instrument


def serializable(value):
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _roots_for_report(report, federation):
    return frozenset().union(*(federation.roots(s) for s in report['sources']))


def coverage_audit(reports, min_side=2, min_known=.7):
    """Expose all reasons the imported candidates() can silently return []."""
    n = len(reports)
    rows = {}
    for name in sorted({k for r in reports for k in r.get('attributes', {})}):
        vals = [r.get('attributes', {}).get(name) for r in reports]
        known = [v for v in vals if v is not None]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in known)
        masks = []
        if numeric:
            levels = sorted(set(known))
            masks = [[v <= (a+b)/2 for v in known] for a,b in zip(levels[:-1],levels[1:])]
        else:
            masks = [[str(v) == str(level) for v in known] for level in sorted(set(map(str, known)))]
        admissible = sum(sum(mask) >= min_side and len(mask)-sum(mask) >= min_side for mask in masks)
        frac = len(known)/n if n else 0.
        reasons = []
        if n < 2*min_side: reasons.append('N_TOO_SMALL')
        if frac < min_known: reasons.append('COVERAGE_TOO_LOW')
        if len(known) < 2*min_side: reasons.append('N_KNOWN_TOO_SMALL')
        if not admissible: reasons.append('NO_TWO_VS_TWO_SPLIT')
        rows[name] = dict(n_known=len(known), coverage=frac, admissible_splits=admissible,
                          eligible=not reasons, reasons=reasons)
    return dict(n=n, n_required=2*min_side, min_known=min_known, attributes=rows,
                empty_means_absence=False)


def run_chain(*, name, model, x_range, params, reports, sources, fixed=None,
              n_x=41, n_p=21, n_perm=2000, seed=6101, alpha=.05,
              measured_axes=(), predictive_inputs_external=False):
    """Run the requested sequence with a callable and attributed public reports.

    report: id, margin, sigma, sources, attributes, validity, locator.
    margin sign must name one common estimand; uncertainty cannot be omitted.
    Root-overlapping reports remain in federation but only a deterministic
    disjoint-root subset goes to GLS and independent-permutation candidates(). This is
    conservative thinning, not a correction for all laboratory correlations.
    """
    cpu0, wall0 = time.process_time(), time.perf_counter()
    mechanism = from_mechanism(model, x_range, params, n_x=n_x, n_p=n_p, fixed=fixed)
    fed = Federation()
    for r in reports:
        if not r.get('sources') or not math.isfinite(r['sigma']) or r['sigma'] <= 0:
            raise ValueError('Each report needs traceable sources and positive explicit uncertainty')
        if r['margin'] == 0:
            continue  # federation has only +/- claims, so do not invent a sign
        fed.add_graph(dict(graph_id=r['id'], label='CURATED-PUBLIC', sources=sources,
                          claims=[dict(id='central-estimate',subject='declared-decision-contrast',
                                       object=name,sign=1 if r['margin']>0 else -1,
                                       validity=r.get('validity',{}), evidence=r['sources'])]))
    selected, omitted, seen = [], [], set()
    for r in reports:
        roots = _roots_for_report(r, fed)
        if roots & seen:
            omitted.append(dict(id=r['id'], reason='ROOT_OVERLAP', roots=sorted(roots)))
        else:
            selected.append(r); seen.update(roots)
    audit = coverage_audit(selected)
    data = candidates(selected,n_perm=n_perm,seed=seed)
    net = MarginNet()
    net.add_sources(sources)
    # A heterogeneous edge is retained as a diagnostic, not certified globally.
    edge = name+':benefit'
    # Two cohorts in one paper are not copies of one common numerical estimate.
    # Feeding both to root-only GLS would falsely impose perfect correlation.
    net.add_edge(edge,['declared-decision-contrast', name], selected)
    est = net.estimate(edge)
    decision = Decision('positive mean intervention benefit: '+name, 'margin',edge=edge,alpha=alpha)
    state = SimpleNamespace(margins=net)
    raw = certify(decision,state)
    instrument = Instrument('independent paired benefit measurement',1.,1.,sigma=est.s if math.isfinite(est.s) else .1)
    attribution = flip_attribution(decision,state,[instrument])
    conflicts = serializable(fed.stress_points())
    match = sorted({r['attribute'] for r in mechanism if r['score']>0} & {c.attribute for c in data if c.p_value <= alpha})
    biological = ('CONDITIONAL_CANDIDATE' if match and set(match)<=set(measured_axes)
                  and predictive_inputs_external and est.kind not in ('CONTRADICTION','REGIME-BOUNDARY','NO-DATA')
                  else 'UNKNOWN')
    return serializable(dict(name=name,mechanism_sweep=mechanism,
         federation=dict(n_reports=len(reports), structural_n_eff=fed.n_eff([s for r in reports for s in r['sources']]),
                         root_sets={r['id']:sorted(_roots_for_report(r,fed)) for r in reports},
                         stress_points=conflicts, signs='central estimates; not assertions of significant effects',
                         statistical_independence='UNKNOWN: common laboratories and participant overlap must be audited'),
         data=dict(selected_reports=[r['id'] for r in selected],omitted=omitted,coverage=audit,candidates=data,
                   permutation_scope='independent selected reports conditional on exchangeability; not a familywise p-value'),
         margin_estimate=est,raw_gaussian_cert=raw,flip_attribution=attribution,
         certificate_report_subset=[r['id'] for r in selected],
         physical_identification=biological,matched_axes=match,
         no_flip_gate='NO_ADDED_SIGN_INFORMATION' if all(r['score']==0 for r in mechanism) else 'SIGN_REVERSAL_PRESENT',
         costs=dict(cpu_s=time.process_time()-cpu0,wall_s=time.perf_counter()-wall0)))


def moment_feasibility(support, mean, sd, sample_n=None):
    """Known Bhatia-Davis bound. A rejection is independent of input distribution.

    Endpoint observation uncertainty is not automatically a population support.
    An attainable pair of moments does not identify a physiological mechanism.
    """
    lo,hi=map(float,support)
    feasible_mean=lo <= mean <= hi
    if sample_n is not None and sample_n < 2:
        raise ValueError('sample SD requires n>=2')
    correction=sample_n/(sample_n-1) if sample_n is not None else 1.
    max_var=correction*(hi-mean)*(mean-lo) if feasible_mean else 0.
    return dict(support=[lo,hi],observed_mean=mean,observed_sd=sd,
                mean_feasible=feasible_mean, variance_upper_bound=max_var,
                sd_upper_bound=math.sqrt(max(0.,max_var)),
                moments_feasible=bool(feasible_mean and sd**2 <= max_var+1e-14),
                sample_n=sample_n,finite_sample_variance_correction=correction,
                identified=False,scope='necessary condition under true support; sample SD corrected; no distribution/covariance specified')


def required_new_measurement(est_m, est_s, sigma_new, alpha=.05):
    """Actual new observed margin that changes Gaussian certify().holds."""
    post_s=(1/est_s**2+1/sigma_new**2)**-.5
    observed_threshold=sigma_new**2*(norm.ppf(1-alpha)/post_s-est_m/est_s**2)
    return dict(observed_margin_threshold=float(observed_threshold),posterior_sd=post_s,
                status_holds_for='new observed margin >= threshold',conditional_error_model='independent Gaussian')


def finite_delivery_certificate(*, content_ratio_bounds, flow_ratio_bounds,
                                same_deficit=True, same_capacity=True, positive_saturation=True):
    """Set certificate, not a posterior. No assumed distribution/independence.

    Given D2/D1=(Q2/Q1)(Ca2/Ca1), positive tau=A/J*(1+K/D),
    the finite tau sign is the opposite delivery sign if A,J,K are fixed.
    Bounds may be conservative projections of correlated joint inputs.
    """
    clo,chi=content_ratio_bounds; flo,fhi=flow_ratio_bounds
    if min(clo,flo)<=0 or chi<clo or fhi<flo:
        raise ValueError('positive ordered ratio bounds required')
    lo,hi=flo*clo,fhi*chi
    valid=same_deficit and same_capacity and positive_saturation
    decision=('FASTER' if lo>1 else 'SLOWER' if hi<1 else 'UNKNOWN') if valid else 'UNKNOWN'
    return dict(delivery_ratio_bounds=[lo,hi],faster_margin=lo-1,slower_margin=1-hi,
                decision=decision,critical_flow_ratio_bounds=[1/chi,1/clo],
                flip_quantity='same-tissue paired Q2/Q1 relative to Ca1/Ca2',
                guarantee='every joint input distribution supported in the declared intervals',
                assumptions=dict(same_deficit=same_deficit,same_capacity=same_capacity,positive_saturation=positive_saturation),
                empirical_scope='requires actual tissue/protocol input validity; no endpoint fitting')
