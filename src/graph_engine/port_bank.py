"""Versioned source ports and explicitly separate Gaussian structural proxies.

The surrogate is not a worker-success model. Directed acquisition is a prerequisite
for offering a target build, even when Gaussian conditioning can infer it inversely.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from .precision_form import Candidate, PrecisionForm
from .typed_throws import Part, Port, closure, set_value_Q_bits
from .next_actions import EngineState, triple_bundle
from .claim_federation import Federation


class PortBank:
    def __init__(self, data, digest):
        self.data, self.sha256 = data, digest
        self.id, self.version = data['id'], data['version']
        if data['schema'] != 'source_port_bank_v1':
            raise ValueError('unsupported bank schema')
        self.quantities = {q['id']: q for q in data['quantities']}
        self.nodes = {n['id']: n for n in data['nodes']}
        if len(self.nodes) != len(data['nodes']) or len(self.quantities) != len(data['quantities']):
            raise ValueError('duplicate canonical ID')
        required = {'inputs', 'outputs', 'and_inputs', 'direction', 'operator',
                    'validity', 'instance', 'provenance', 'cost', 'leaf_status'}
        for n in self.nodes.values():
            if not required <= n.keys() or n['direction'] != 'forward':
                raise ValueError('incomplete directed node')
            if n['and_inputs'] != [p['q'] for p in n['inputs'] if p['required']]:
                raise ValueError('AND list differs from required ports')
            if not math.isfinite(n['cost']) or n['cost'] <= 0:
                raise ValueError('invalid node cost')
            if len(n['outputs']) != 1:
                raise ValueError('this bank version requires one output/observation per node')
            for p in n['inputs'] + n['outputs']:
                q = self.quantities[p['q']]
                if any(p[k] != q[k] for k in ('unit', 'kind', 'instance')):
                    raise ValueError('port type or instance differs from canonical quantity')
            for lo, hi in n['validity'].values():
                if not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
                    raise ValueError('invalid validity box')
            if not n['provenance']:
                raise ValueError('missing provenance')
        self.given = set(data['given'])
        if not self.given <= self.quantities.keys() or data['target'] not in self.quantities:
            raise ValueError('unknown given or target')
        g = data['gaussian']
        if g['semantics'] != 'uncalibrated_dimensionless_artifact_proxy':
            raise ValueError('source-port Gaussian image must be labelled as a surrogate')
        self.coords = g['coordinates']
        if len(set(self.coords)) != len(self.coords) or set(self.coords) != self.quantities.keys():
            raise ValueError('Gaussian coordinates differ from bank quantities')
        self.index = {q: i for i, q in enumerate(self.coords)}
        self.J = np.array(g['prior_precision'], dtype=float)
        self.H = np.array([g['observations'][n]['row'] for n in self.nodes], dtype=float)
        self.sigmas = np.array([g['observations'][n]['sigma'] for n in self.nodes], dtype=float)
        self.Q = np.array(g['Q'], dtype=float)
        d = len(self.coords)
        if self.J.shape != (d, d) or self.H.shape != (len(self.nodes), d) or self.Q.shape != (1, d):
            raise ValueError('incomplete Gaussian model')
        if not all(np.all(np.isfinite(x)) for x in (self.J, self.H, self.sigmas, self.Q)):
            raise ValueError('nonfinite Gaussian data')
        if np.any(self.sigmas <= 0) or not np.allclose(self.J, self.J.T):
            raise ValueError('invalid precision or noise')
        np.linalg.cholesky(self.J)
        # This is a declared structural model, never a sum of dimensioned physical quantities.
        for k, n in enumerate(self.nodes.values()):
            h = np.zeros(d)
            h[self.index[n['outputs'][0]['q']]] = 1
            for p in n['inputs']:
                if p['required']:
                    h[self.index[p['q']]] -= 1
            if not np.array_equal(self.H[k], h):
                raise ValueError('observation differs from declared structural ports')
        expected_q = np.zeros((1, d)); expected_q[0, self.index[data['target']]] = 1
        if not np.array_equal(self.Q, expected_q):
            raise ValueError('Q differs from declared target')
        self.positions = {n: i for i, n in enumerate(self.nodes)}
        self.parts = {n['id']: Part(
            id=n['id'], report=n['id'], family='graph', does=n['operator'],
            inputs=[Port(p['q'], p['unit'], p['kind'], p['required'], p['q'] in self.given) for p in n['inputs']],
            outputs=[Port(p['q'], p['unit'], p['kind']) for p in n['outputs']],
            operator=n['operator_tokens'], equation=n['equation'],
            assumptions=n['assumptions'], box=n['validity'], reach=n['reach'],
            status='reported', source=n['provenance'][0]['path']) for n in self.nodes.values()}

    @classmethod
    def load(cls, path, expected_sha256=None):
        raw = Path(path).read_bytes(); digest = sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('bank SHA256 mismatch')
        return cls(json.loads(raw), digest)

    def reference(self):
        return dict(id=self.id, version=self.version, sha256=self.sha256)

    def form(self):
        return PrecisionForm(self.J.copy(), np.zeros(len(self.coords)))

    def candidates(self, ids=None):
        return [Candidate(n, self.H[self.positions[n]], self.sigmas[self.positions[n]], self.nodes[n]['cost'])
                for n in (self.nodes if ids is None else ids)]

    def execution(self, ids):
        """Forward-only AND acquisition; signed/physical validity is never inferred from prose."""
        ids = tuple(ids)
        if len(set(ids)) != len(ids) or any(n not in self.nodes for n in ids):
            raise ValueError('invalid selected members')
        # Whole-set common declared pilot domain must be nonempty.
        box = {}
        for n in ids:
            for axis, (lo, hi) in self.nodes[n]['validity'].items():
                a, b = box.get(axis, (lo, hi)); box[axis] = (max(a, lo), min(b, hi))
        if any(lo > hi for lo, hi in box.values()):
            return set(self.given), (), {n: ['incompatible validity'] for n in ids}
        available, order = set(self.given), []
        pending = list(ids)
        while pending:
            ready = [n for n in pending if set(self.nodes[n]['and_inputs']) <= available]
            if not ready:
                break
            for n in ready:
                available.update(p['q'] for p in self.nodes[n]['outputs'])
                order.append(n); pending.remove(n)
        return available, tuple(order), {n: sorted(set(self.nodes[n]['and_inputs']) - available) for n in pending}

    def value(self, ids, acquisition=False):
        ids = tuple(ids)
        if acquisition:
            available, order, _ = self.execution(ids)
            if self.data['target'] not in available:
                return 0.0
            ids = order
        ks = [self.positions[n] for n in ids]
        return 0.0 if not ks else set_value_Q_bits(self.form(), self.H[ks], self.sigmas[ks], self.Q)

    def best_executable_set(self, ids, k=3, max_sets=100000):
        ids = tuple(ids)
        if math.comb(len(ids), k) > max_sets:
            raise ValueError('acquisition selection exceeds max_sets')
        sets = list(combinations(ids, k))
        eligible = [s for s in sets if self.data['target'] in self.execution(s)[0]]
        if not eligible:
            return None
        return max(eligible, key=lambda s: self.value(s, acquisition=True) / sum(self.nodes[n]['cost'] for n in s))

    def native_closure(self, ids):
        return closure(self.parts, ids, self.given)

    def action(self, ids):
        """Offer a source-port build in structural currency, with its bank reference."""
        if self.data['target'] not in self.execution(ids)[0]:
            raise ValueError('target has no executable port closure')
        state = EngineState(typed_parts=self.parts, triple_given=self.given,
                            triple_costs={n: d['cost'] for n, d in self.nodes.items()})
        a = triple_bundle(state, ids, self.data['target'], valuation='closure')
        a.meta['bank'] = self.reference()
        a.meta['empirical_success_calibration'] = 'UNKNOWN'
        return a

    def worker_graph(self, export):
        """Validate before Federation.add_graph (which otherwise accepts sign=0 as negative)."""
        if export['bank'] != self.reference():
            raise ValueError('worker bank ID/version/hash mismatch')
        members = export['members']
        if len(members) != 3 or len(set(members)) != 3 or not set(members) <= self.nodes.keys():
            raise ValueError('invalid worker triple')
        out = {'graph_id': export['graph_id'], 'sources': export.get('sources', []), 'claims': []}
        seen = set()
        for e in export.get('signed_edges', []):
            if e['id'] in seen:
                raise ValueError('duplicate signed claim ID')
            seen.add(e['id'])
            if type(e['sign']) is not int or e['sign'] not in (-1, 1):
                raise ValueError('explicit sign must be -1 or +1')
            domain = self.data['sign_domains'][e['domain']]
            if e['semantics'] != domain['semantics'] or e['instance'] != domain['instance']:
                raise ValueError('signed composition domain/instance mismatch')
            if e['subject'] not in domain['variables'] or e['object'] not in domain['variables']:
                raise ValueError('signed endpoint not in domain')
            if set(e['validity']) != set(domain['validity']):
                raise ValueError('explicit common validity required')
            for axis, (lo, hi) in e['validity'].items():
                a, b = domain['validity'][axis]
                if not all(math.isfinite(x) for x in (lo, hi)) or not a <= lo <= hi <= b:
                    raise ValueError('signed claim outside bank domain')
            if not e.get('evidence'):
                raise ValueError('signed claim missing evidence')
            out['claims'].append({k: e[k] for k in ('id', 'subject', 'object', 'sign', 'instance', 'validity', 'evidence')})
        return out

    def federation(self, export):
        f = Federation(); f.add_graph(self.worker_graph(export)); return f
