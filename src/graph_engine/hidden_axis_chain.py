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
from .claim_types import deductive_certificate
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


def _no_flip_gate(mechanism, joint):
    """The sign verdict, kept honest about which sweep earned it.

    `from_mechanism` is one-at-a-time, so all-zero scores clear only the midpoint slice. The strong
    statement needs the joint scan over the parameter product, which `claim_types.deductive_certificate`
    already performs with the corners always included. A diagonal flip surface is exactly the case the
    axis-aligned sweep misses and the joint scan catches, so the two verdicts are kept apart.
    """
    if any(r['score'] > 0 for r in mechanism):
        return 'SIGN_REVERSAL_PRESENT'
    if not joint.get('ok'):
        # A zero slope and an opposite slope are different findings. The mechanism declaring no sign at a
        # point leaves the quantity locally uninformative; the mechanism running backwards is what forces
        # the axis into the validity box. The scan's first witness is often the zero even when a strict
        # reversal exists elsewhere in the same box, so the count decides the verdict, not the witness.
        if joint.get('n_opposite_sign'):
            return 'SIGN_REVERSED_IN_JOINT_BOX'
        return 'SIGN_DEGENERATE_IN_JOINT_BOX'
    return 'NO_ADDED_SIGN_INFORMATION'


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
    # from_mechanism is one-at-a-time, so all-zero scores mean "free on the midpoint slice" and not
    # "free on the box": a flip surface diagonal in two parameters is missed by the axis-aligned cross.
    # The corners are where such a surface is crossed first, so the strong gate is not earned without them.
    mid = {k: (fixed or {}).get(k, (lo + hi) / 2) for k, (lo, hi) in params.items()}
    hh = (x_range[1] - x_range[0]) / (n_x * 10)
    xm = (x_range[0] + x_range[1]) / 2
    mid_sign = 1 if model(xm + hh, **mid) - model(xm - hh, **mid) >= 0 else -1
    joint = deductive_certificate(model, x_range, params, sign=mid_sign, n_x=n_x)
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
         no_flip_gate=_no_flip_gate(mechanism, joint), joint_box_certificate=joint, midpoint_sign=mid_sign,
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


def instrument_can_decide(*, est_m, est_s, sigma_new, support, decision_band, alpha=.05,
                          declared_domain=None, sampled_domain=None, sigma_source='declared'):
    """Can ANY reading of this instrument certify the decision, before the search is run?

    Two refusals, and they are different failures that must not be collapsed:

    PRECISION. `required_new_measurement` gives the observation that would have to be seen for
    certify() to hold. If that observation lies outside the quantity's declared support, no reading
    of this instrument can produce it, and repeating the measurement does not help -- the limit is
    the instrument, not the sample size. Threshold-free: the verdict is inside-or-outside a declared
    interval, never a ratio compared to a cutoff. `resolution_ratio` is reported as a number for the
    reader, and nothing is decided by it.

    COVERAGE. An instrument that reads only part of the domain the decision is declared over cannot
    certify the declared domain however precise each reading is. This is the quantifier gap -- the
    evaluated set is a proper subset of the declared one -- and it is not a precision problem. Pass
    `declared_domain` and `sampled_domain` as comparable sets to have it checked; omit them and the
    coverage question is reported as UNCHECKED rather than passed, because an unasked question is
    not an answered one.

    `sigma_required` names the instrument that WOULD decide it: the sigma at which the required
    observation re-enters the support, found by bisection. That is what makes the refusal actionable
    instead of a verdict of impossibility.

    With no prior information the required observation reduces to exactly sigma * z_(1-alpha): the
    bisection is only needed when a prior narrows it. So the precision refusal fires precisely when
    sigma * z_(1-alpha) exceeds the support's upper bound, which is a criterion a reader can apply by
    hand. A wide support therefore makes this condition non-binding, and that is a property of the
    declaration rather than of the instrument -- measured on a case with sigma 3.96 and support
    [0, 20], the precision test does not fire while the coverage test does.

    `sigma_source` is carried through and reported. A sigma derived from a published R-squared is not
    a reported residual, and a test built on one should say so rather than present it as measured.
    """
    lo, hi = map(float, support)
    if hi < lo:
        raise ValueError('support must be ordered')
    band_lo, band_hi = map(float, decision_band)
    if band_hi < band_lo or band_lo <= 0:
        raise ValueError('decision_band must be positive and ordered')

    def required(sigma):
        return required_new_measurement(est_m, est_s, sigma, alpha=alpha)['observed_margin_threshold']

    req = required(sigma_new)
    in_support = lo <= req <= hi
    reasons = []
    if not in_support:
        reasons.append('REQUIRED_OBSERVATION_OUTSIDE_SUPPORT')

    # The sigma that would bring the requirement back inside the support. The requirement tends to 0
    # as sigma does, so a bracket exists whenever the support contains 0 or a value below req.
    sigma_required = None
    if not in_support:
        a, b = sigma_new * 1e-9, float(sigma_new)
        if lo <= required(a) <= hi:
            for _ in range(200):
                mid = math.sqrt(a * b)
                if lo <= required(mid) <= hi:
                    a = mid
                else:
                    b = mid
            sigma_required = a

    if declared_domain is None or sampled_domain is None:
        coverage = 'UNCHECKED'
    else:
        declared, sampled = set(declared_domain), set(sampled_domain)
        if not sampled <= declared:
            raise ValueError('sampled_domain must be a subset of declared_domain')
        coverage = 'FULL' if sampled == declared else 'SUBSET_OF_DECLARED_DOMAIN'
        if coverage != 'FULL':
            reasons.append('INSTRUMENT_SAMPLES_A_SUBSET_OF_THE_DECLARED_DOMAIN')

    return dict(required_observation=float(req), support=[lo, hi],
                required_observation_in_support=bool(in_support),
                resolution_ratio=float(sigma_new) / band_lo,
                decision_band=[band_lo, band_hi], sigma_new=float(sigma_new),
                sigma_required=sigma_required, sigma_source=sigma_source,
                coverage=coverage, refusal_reasons=reasons,
                verdict='CAN_DECIDE' if not reasons else 'CANNOT_DECIDE',
                scope='necessary conditions on the instrument alone; says nothing about whether the '
                      'quantity is what the decision needs, nor about any biology or mechanism')
