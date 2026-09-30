"""Content-addressed history layer. Never edits an input or native graph.

All scientific dependency/ranking operations stay in the pinned Graph Engine.
The relations here are documentary provenance, never proof dependencies.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import random
import re
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = Path('/home/anton/research/inference_training_20260921')
PILOT = RESEARCH / 'jobs/research_graph_pilot'
DISTILL = Path('/home/anton/research/FREE_AUTONOMY_20260926/ASTRA_DISTILL_20260928')
PROGRAM = Path('/home/anton/projects/3fold-workspaces/bodytwin/tasks/free48/SEED_PROGRAM_500_IMPROVED_20260929.json')
STORE = Path('/mnt/games-240/research/claude24h_night/G5_ADMISSION_TRIPLE/evidence')
TOKEN = re.compile(r'(?<![A-Za-z0-9_/-])[A-Za-z0-9][A-Za-z0-9_/-]*(?:\:[0-9]+)?(?![A-Za-z0-9_/-])')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def encoded(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def dump(path, x):
    path = Path(path)
    # Stream the large layer directly to the designated disk; lane link is the API.
    target = STORE.parent / path.name if path.name == 'BANK_LAYER.json' else path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(x, f, ensure_ascii=False, indent=1)
        f.write('\n')
    if tmp.stat().st_size > 50_000_000 and not str(target).startswith('/mnt/games-240/research/claude24h_night/F1_FOLD_BANK/'):
        big = STORE.parent / path.name
        os.replace(tmp, big)
        target = big
    else:
        os.replace(tmp, target)
    if target != path.absolute() and not path.is_symlink():
        link = path.with_suffix(path.suffix + '.linktmp')
        link.symlink_to(target)
        os.replace(link, path)


def pointer(obj, ptr):
    for k in ptr.strip('/').split('/') if ptr else []:
        k = k.replace('~1', '/').replace('~0', '~')
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


class Builder:
    def __init__(self):
        self.nodes = {}
        self.docs = {}
        self.raw = {}
        self.edges = {}
        self.alias_ids = collections.defaultdict(set)
        self.paths = {}
        self.unresolved = []
        self.occurrences = []
        self.card_bindings = {}
        self.missing_paths = set()
        self.relocations = {}
        recovery = HERE / 'RECOVERED_PATHS.json'
        if recovery.is_file():
            self.relocations = json.loads(recovery.read_text())['mapping']
            self.document(recovery)

    def document(self, path):
        path = str(path)
        if path in self.docs:
            return self.docs[path]
        if path in self.missing_paths:
            return None
        try:
            raw = Path(path).read_bytes()
        except (OSError, ValueError) as e:
            self.unresolved.append({'kind': 'missing_document', 'path': path, 'error': str(e)})
            self.missing_paths.add(path)
            return None
        h = sha(raw)
        STORE.mkdir(parents=True, exist_ok=True)
        snap = STORE / h
        if not snap.exists():
            snap.write_bytes(raw)
        elif sha(snap.read_bytes()) != h:
            raise ValueError('corrupt content-addressed snapshot')
        doc = {'path': path, 'sha256': h, 'bytes': len(raw), 'snapshot': str(snap)}
        self.docs[path] = doc
        self.raw[h] = raw
        return doc

    def witness(self, doc, **kw):
        return {'document': doc['path'], 'sha256': doc['sha256'], **kw}

    def report(self, occurrence, family=None, indexed_sha=None, kind='report'):
        declared_path = occurrence['path']
        if not Path(declared_path).is_file() and declared_path in self.relocations:
            recovery = self.relocations[declared_path]
            occurrence = {**occurrence, 'path': recovery['recovered_path'], 'indexed_path': declared_path}
        doc = self.document(occurrence['path'])
        if doc is None:
            return None
        h = doc['sha256']
        nid = 'B-' + h
        fam = family or occurrence.get('family', 'UNKNOWN')
        if nid not in self.nodes:
            text = self.raw[h].decode('utf-8', errors='replace')
            title = next((l.strip('# *') for l in text.splitlines() if l.strip()), occurrence['id'])
            self.nodes[nid] = {'id': nid, 'kind': kind, 'status': 'PENDING',
                'producer': occurrence['id'], 'sha256': h, 'hash_kind': 'file_bytes',
                'family': fam, 'source_family': [fam], 'target': [],
                'goal': None,
                'time': occurrence.get('time', Path(occurrence['path']).stat().st_mtime),
                'time_kind': 'index_timestamp_or_file_mtime_not_execution_time',
                'title': title[:500], 'excerpt': text[:1800], 'aliases': [], 'reviews': [],
                'scientific_acceptance': False, 'root_verified': False}
        node = self.nodes[nid]
        alias = {k: occurrence[k] for k in ('id', 'path', 'indexed_path', 'time', 'origin') if k in occurrence}
        alias['family'] = fam
        if alias not in node['aliases']:
            node['aliases'].append(alias)
        if fam not in node['source_family']:
            node['source_family'].append(fam)
        self.alias_ids[occurrence['id']].add(nid)
        self.paths[occurrence['path']] = nid
        if indexed_sha is not None:
            self.occurrences.append({'id': occurrence['id'], 'path': declared_path, 'resolved_path': occurrence['path'],
                'indexed_sha256': indexed_sha, 'actual_sha256': h, 'node': nid,
                'matches_index': indexed_sha == h})
        return nid

    def resolve(self, rid, h=None, external=False):
        found = self.alias_ids.get(rid, set())
        if h and 'B-' + h in found:
            return 'B-' + h
        if h and 'B-' + h in self.nodes and not found:
            self.unresolved.append({'kind': 'unbound_id_hash', 'reference': rid, 'hash': h})
            return None
        if h and found:
            self.unresolved.append({'kind': 'hash_id_conflict', 'reference': rid, 'hash': h, 'choices': sorted(found)})
            return None
        if len(found) == 1:
            return next(iter(found))
        if len(found) > 1:
            self.unresolved.append({'kind': 'ambiguous_id', 'reference': rid, 'choices': sorted(found)})
            return None
        if not external:
            self.unresolved.append({'kind': 'unresolved_id', 'reference': rid})
            return None
        nid = 'REF-' + sha(rid.encode())
        self.nodes.setdefault(nid, {'id': nid, 'kind': 'source_reference', 'status': 'PENDING',
            'producer': 'UNKNOWN', 'sha256': sha(rid.encode()), 'hash_kind': 'identifier_only',
            'family': 'external_reference', 'source_family': ['UNKNOWN'], 'target': [], 'time': None,
            'title': rid, 'aliases': [{'id': rid}], 'reviews': [],
            'scientific_acceptance': False, 'root_verified': False})
        return nid

    def edge(self, a, b, typ, evidence):
        if not a or not b or a == b:
            return
        key = (a, b, typ)
        row = self.edges.setdefault(key, {'id': 'E-' + sha(encoded(key)),
            'from': a, 'to': b, 'type': typ, 'evidence': [], 'scientific_dependency': False})
        if evidence not in row['evidence']:
            row['evidence'].append(evidence)

    def parents(self, nid, obj, doc, base=''):
        for key in ('after', 'parent', 'parents', 'source_ids', 'parent_ids'):
            vals = obj.get(key, [])
            vals = [vals] if isinstance(vals, str) else vals
            if not isinstance(vals, list):
                continue
            for i, rid in enumerate(vals):
                if isinstance(rid, dict):
                    rid = rid.get('id')
                if not isinstance(rid, str) or not rid:
                    continue
                w = self.witness(doc, pointer=base + '/' + key + (f'/{i}' if isinstance(obj[key], list) else ''), reference=rid)
                self.edge(self.resolve(rid, external=True), nid, 'parent', w)

    def citations(self, nid, doc):
        text = self.raw[doc['sha256']].decode('utf-8', errors='replace')
        seen = set()
        for line, s in enumerate(text.splitlines(), 1):
            for m in TOKEN.finditer(s):
                rid = m.group()
                # Short M1 etc are often metric labels, not mutation identity.
                if rid in seen or re.fullmatch(r'M\d+', rid) or rid not in self.alias_ids:
                    continue
                seen.add(rid)
                dst = self.resolve(rid)
                self.edge(nid, dst, 'cited_id', self.witness(doc, line=line, reference=rid, quote=s[:800]))

    def metadata(self, nid, report_alias):
        folder = Path(report_alias['path']).parent
        jobpath = folder / 'JOB.json'
        if not jobpath.is_file():
            return
        doc = self.document(jobpath)
        try:
            job = json.loads(self.raw[doc['sha256']])
        except (ValueError, TypeError):
            return
        self.parents(nid, job, doc)
        n = self.nodes[nid]
        if job.get('target_id') and job['target_id'] not in n['target']:
            n['target'].append(job['target_id'])
        if job.get('decision') and not n.get('goal'):
            n['goal'] = job['decision']
        # Actual seed job identity, never a bare 'M1' in narrative.
        jid = job.get('id', report_alias['id'])
        m = re.fullmatch(r'BT-FW48-SEED-(\d{3})', jid)
        if m:
            seed = self.resolve('M' + str(int(m[1])))
            self.edge(seed, nid, 'same_m_id', self.witness(doc, pointer='/id', reference=jid))
        catpath = folder / 'inputs/SOURCE_CATALOG.json'
        if not catpath.is_file():
            return
        catdoc = self.document(catpath)
        try:
            cat = json.loads(self.raw[catdoc['sha256']])
        except ValueError:
            return
        for key in job.get('source_keys', []):
            item = cat.get(key)
            if not isinstance(item, dict):
                continue
            ptr = '/' + key.replace('~', '~0').replace('/', '~1')
            self.parents(nid, item, catdoc, ptr)
            # Evidence means supplied resource in declared source packet, not executed code.
            for i, rel in enumerate(item.get('files', [])):
                if not isinstance(rel, str) or Path(rel).suffix not in {'.py', '.json', '.csv', '.tsv', '.npz', '.lean'}:
                    continue
                p = folder / 'inputs' / key / rel
                if not p.is_file():
                    continue
                resdoc = self.document(p)
                rid = 'RES-' + resdoc['sha256']
                self.nodes.setdefault(rid, {'id': rid, 'kind': 'resource', 'status': 'PENDING',
                    'sha256': resdoc['sha256'], 'hash_kind': 'file_bytes', 'producer': 'UNKNOWN',
                    'family': 'supplied_code_data', 'source_family': [], 'target': [], 'time': None,
                    'title': rel, 'aliases': [], 'reviews': [], 'scientific_acceptance': False, 'root_verified': False})
                if str(p) not in self.nodes[rid]['aliases']:
                    self.nodes[rid]['aliases'].append(str(p))
                self.edge(nid, rid, 'shared_code_data', self.witness(catdoc,
                    pointer=ptr + f'/files/{i}', reference=rel, resource_document=str(p),
                    resource_sha256=resdoc['sha256'], source_key=key, job_document=doc['path']))

    def cards(self, cards, doc, prefix='/cards'):
        for i, card in enumerate(cards):
            if not card.get('producer_report'):
                continue
            nid = self.report({'id': card.get('job', card['id']), 'path': card['producer_report']}, family='graph_combinatorics')
            if not nid:
                nid = self.missing_report(card, doc, f'{prefix}/{i}')
                if not nid:continue
            self.card_bindings[card['id']] = nid
            self.alias_ids[card['id']].add(nid)
            self.nodes[nid].setdefault('card_ids', []).append(card['id'])
            self.nodes[nid].setdefault('original_cards', []).append(card)
            for j, src in enumerate(card.get('sources', [])):
                dst = self.resolve(src['id'], src.get('sha256'))
                self.edge(dst, nid, 'parent', self.witness(doc, pointer=f'{prefix}/{i}/sources/{j}', reference=src['id'], declared_hash=src.get('sha256')))

    def missing_report(self, card, doc, ptr):
        h=card.get('producer_report_sha256')
        if not h:return None
        nid='B-'+h
        self.nodes.setdefault(nid, {'id':nid, 'kind':'missing_report', 'status':'PENDING',
            'producer':card['job'],'sha256':h,'hash_kind':'declared_report_sha256_unavailable_bytes',
            'family':'graph_combinatorics','source_family':['graph_combinatorics'],
            'target':[],'time':None,'title':card['title'],'excerpt':card.get('result_summary',''),
            'aliases':[{'id':card['job'],'path':card['producer_report']}], 'reviews':[],
            'scientific_acceptance':False,'root_verified':False,'content_available':False,
            'definition_evidence':self.witness(doc,pointer=ptr)})
        self.alias_ids[card['job']].add(nid)
        return nid

    def review(self, doc, authority, node=None, scope=None, verdict=None, locator=None):
        if not doc:
            return
        refs = []
        text = self.raw[doc['sha256']].decode('utf-8', errors='replace')
        if node:
            refs = [node]
        else:
            for m in TOKEN.finditer(text):
                rid = m.group()
                if rid in self.alias_ids and not re.fullmatch(r'M\d+', rid):
                    dst = self.resolve(rid)
                    if dst:
                        refs.append(dst)
        for dst in sorted(set(refs)):
            # Records scoped review occurrence, never promotes a whole report's claims.
            row = {'authority': authority, 'document': doc['path'], 'sha256': doc['sha256'],
                'scope': scope or 'Document mentions this exact artifact ID; per-claim verdict must be read.',
                'verdict': verdict or 'REVIEW_DOCUMENT_MENTION', 'scientific_acceptance': False,
                'binding': 'exact_result_hash' if node else 'mention_only', 'locator': locator}
            self.nodes[dst]['reviews'].append(row)
            if node and verdict:
                self.nodes[dst]['status'] = 'REVIEWED_SCOPED'

    def build(self, limit=None):
        started = time.perf_counter()
        idxdoc = self.document(DISTILL / 'RESEARCH_INDEX.json')
        index = json.loads(self.raw[idxdoc['sha256']])
        seedoc = self.document(PROGRAM)
        seeds = json.loads(self.raw[seedoc['sha256']])
        for i, row in enumerate(seeds):
            nid = 'M-' + row['id'] + '-' + sha(encoded(row))
            self.nodes[nid] = {'id': nid, 'kind': 'm_job', 'producer': row['job_id'],
                'status': 'PENDING', 'sha256': sha(encoded(row)), 'hash_kind': 'canonical_json_row',
                'family': row['theme'], 'source_family': ['seed_program_500'],
                'target': row.get('graph_grounding', {}).get('working_packet_ids', []),
                'goal': row['decision'], 'title': row['twist'], 'time': PROGRAM.stat().st_mtime,
                'time_kind': 'program_file_mtime_not_execution_time',
                'aliases': [{'id': row['id']}, {'id': row['job_id']}], 'reviews': [],
                'scientific_acceptance': False, 'root_verified': False,
                'definition_evidence': self.witness(seedoc, pointer=f'/{i}')}
            self.alias_ids[row['id']].add(nid)
            # Separate program definition from executed report of that job.
            self.nodes[nid]['job_id'] = row['job_id']
        rows = index['reports'][:limit] if limit else index['reports']
        for row in rows:
            for occ in [row['canonical'], *row.get('aliases', [])]:
                self.report(occ, row['canonical']['family'], row['sha256'])
        report_ids_before = set(self.nodes)
        cardsdoc = self.document(DISTILL / 'GRAPH_COMBINATORICS/ACTIVE/CARDS_ALL.json')
        cards = json.loads(self.raw[cardsdoc['sha256']])['cards']
        self.cards(cards, cardsdoc)
        seldoc = self.document(DISTILL / 'GRAPH_COMBINATORICS/ACTIVE/SELECTION.json')
        sel = json.loads(self.raw[seldoc['sha256']])
        for i, candidate in enumerate(sel.get('candidates', [])):
            endpoints = []
            for key in ('new_card', 'old_card'):
                card = candidate[key]
                nid = self.report({'id': card['job'], 'path': card['producer_report']}, family='graph_combinatorics')
                if not nid:nid=self.missing_report(card,seldoc,f'/candidates/{i}/{key}')
                self.card_bindings[card['id']] = nid
                self.alias_ids[card['id']].add(nid)
                if nid:self.nodes[nid].setdefault('card_ids', []).append(card['id'])
                endpoints.append(nid)
            self.edge(*endpoints, 'combination_candidate', self.witness(seldoc, pointer=f'/candidates/{i}',
                reference=[candidate['new_card']['id'], candidate['old_card']['id']]))
        feedoc = self.document(DISTILL / 'GRAPH_COMBINATORICS/CLOUD/FEEDBACK_INDEX.json')
        feedback = json.loads(self.raw[feedoc['sha256']])
        for row in feedback.get('results', []):
            if row.get('report_path'):
                self.report({'id': row['job'], 'path': row['report_path']}, family='graph_cloud_feedback')
        # Report versions that actually exist for the 500 historical job IDs.
        for row in seeds:
            for root in [Path('/home/anton/projects/3fold-workspaces/bodytwin/results'),
                         Path('/mnt/games-240/research/bunny48_20260926/bodytwin')]:
                p = root / row['job_id'] / 'RESULTS.md'
                if p.is_file():
                    self.report({'id':row['job_id'],'path':str(p)},family='seed_program_execution_history')
                    break
        # All IDs are known before resolution. Metadata is restricted to report folders.
        for nid, n in list(self.nodes.items()):
            if n['kind'] == 'report':
                for alias in n['aliases']:
                    self.metadata(nid, alias)
                self.citations(nid, self.docs[n['aliases'][0]['path']])
        # Same M id via exact executed-job alias; v6 definition is NOT earlier result.
        seed_rows = {r['id']:r for r in seeds}
        for n in list(self.nodes.values()):
            if n['kind'] == 'm_job':
                for dst in self.alias_ids.get(n['job_id'], []):
                    self.edge(n['id'], dst, 'same_m_id', {**n['definition_evidence'], 'reference': n['job_id']})
                row = seed_rows[next(a['id'] for a in n['aliases'] if re.fullmatch(r'M\d+',a['id']))]
                for j, bridge in enumerate(row.get('innovation', {}).get('method_bridges', [])):
                    dst = self.resolve(bridge['mutation_id'])
                    typ = 'discriminating_test_candidate' if bridge.get('relation') == 'PROPOSED_DISCRIMINATING_TEST_CONSUMER' else 'combination_candidate'
                    self.edge(n['id'], dst, typ, self.witness(seedoc,
                        pointer=n['definition_evidence']['pointer'] + f'/innovation/method_bridges/{j}', reference=bridge['mutation_id']))
        # Shared resources are represented by incidence rather than an O(d^2) clique.
        users = collections.defaultdict(set)
        for e in self.edges.values():
            if e['type'] == 'shared_code_data':
                users[e['to']].add(e['from'])
        private = {rid for rid, uses in users.items() if len(uses) < 2}
        self.edges = {k:e for k,e in self.edges.items() if e['to'] not in private}
        for rid in private:
            del self.nodes[rid]
        graphdoc = self.document(PILOT / 'GRAPH.json')
        self.document(PILOT / 'LEDGER.jsonl')
        root_graph = json.loads(self.raw[graphdoc['sha256']])
        root_ids = {n['id'] for n in root_graph['nodes']}
        # Reviews are evidence sources, not another graph or self-admission.
        for path in sorted((RESEARCH / 'claude_24h/root_review').glob('*/RESULTS.md')):
            doc = self.document(path)
            self.review(doc, 'root_review')
        ledgerdoc = self.docs[str(PILOT / 'LEDGER.jsonl')]
        for i, line in enumerate(self.raw[ledgerdoc['sha256']].decode().splitlines(), 1):
            row = json.loads(line)
            if not row.get('node', '').startswith('X-') or not row.get('artifact'):
                continue
            p = row['artifact']['path']
            doc = self.document(p)
            if not doc or doc['sha256'] != row['artifact']['sha256']:
                continue
            if '/root_review/' in p:
                self.review(doc, 'graphctl_x_ledger', scope=row['claim'], verdict=row['verdict'], locator={'ledger_line': i, 'fold': row['id']})
                for dst in self.nodes.values():
                    if any(r['document'] == p for r in dst.get('reviews', [])):
                        self.edge(row['node'], dst['id'], 'root_review_link', self.witness(ledgerdoc,
                            line=i, reference=row['node'], review_document=p, review_sha256=doc['sha256']))
        # Exact paths supplied in source tree, plus named review directories (read only).
        review_roots = [DISTILL / 'COORDINATOR_REVIEWS',
            Path('/home/anton/projects/3fold-workspaces/bodytwin/COORDINATOR_REVIEWS'),
            Path('/home/anton/projects/3fold-workspaces/bodytwin/results/COORDINATOR_REVIEWS'),
            Path('/home/anton/research/FREE_AUTONOMY_20260926/COORDINATOR_REVIEWS')]
        for r in review_roots:
            if r.is_dir():
                for p in sorted(r.rglob('*')):
                    if p.is_file() and p.suffix in {'.md', '.json'}:
                        self.review(self.document(p), 'COORDINATOR_REVIEWS')
        coordinator_file = Path('/home/anton/research/sol6_recovery_20260923/GRAPH_USE_AUDIT_20260923/COORDINATOR_REVIEWS.json')
        if coordinator_file.is_file():
            codoc = self.document(coordinator_file)
            for i,r in enumerate(json.loads(self.raw[codoc['sha256']]).get('reviews', [])):
                nid = 'B-' + r['result_sha256']
                if nid in self.nodes:
                    self.review(codoc, 'COORDINATOR_REVIEWS', node=nid, scope=r['review_scope'],
                        verdict='SCOPED_EXISTING_COORDINATOR_REVIEW', locator={'pointer':f'/reviews/{i}'})
                else:
                    self.unresolved.append({'kind':'coordinator_result_not_in_report_bank','result_sha256':r['result_sha256'],
                        'document':str(coordinator_file),'scope':r['review_scope']})
        conflicts = {k:sorted(v) for k,v in self.alias_ids.items() if len(v)>1}
        for n in self.nodes.values():
            n.setdefault('goal', None)
        out = {'schema': 'research_bank_layer_v1', 'native_graph': str(PILOT / 'GRAPH.json'),
            'native_graph_sha256': graphdoc['sha256'], 'root_ids': sorted(root_ids),
            'semantics': {'parent': 'declared ancestry, never proof dependency',
                'shared_code_data': 'report → byte-identical shared supplied resource incidence; not execution',
                'same_m_id': 'versioned M definition → prior report; not evidence of v6 execution',
                'discriminating_test_candidate': 'proposed test consumer, not a proposed method combination',
                'REVIEWED_SCOPED': 'existing independent review mentions exact ID; whole-report acceptance false'},
            'nodes': list(self.nodes.values()), 'edges': list(self.edges.values()),
            'documents': list(self.docs.values()), 'occurrences': self.occurrences,
            'id_conflicts': conflicts, 'unresolved': self.unresolved,
            'input_index_counts': {k:index[k] for k in ('report_occurrences','unique_report_contents','exact_copy_aliases')},
            'review_roots_checked': {str(r):r.is_dir() for r in review_roots},
            'cost': {'build_wall_s': time.perf_counter()-started, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'distinct_document_bytes': sum(len(v) for v in self.raw.values()), 'gpu_s': 0},
            'pilot_limit': limit}
        return out


def components(ids, pairs):
    parent = {x:x for x in ids}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a,b in pairs:
        if a in parent and b in parent:
            parent[root(a)] = root(b)
    sizes = collections.Counter(root(x) for x in ids)
    return {'components': len(sizes), 'largest': max(sizes.values(), default=0)}


def metrics(layer):
    graphdoc = next(d for d in layer['documents'] if d['path'] == layer['native_graph'])
    graph = json.loads(Path(graphdoc['snapshot']).read_bytes())
    roots = {n['id'] for n in graph['nodes']}
    old_pairs = [(d,n['id']) for n in graph['nodes'] for d in n['depends_on']]
    ids = roots | {n['id'] for n in layer['nodes']}
    nodes = {n['id']:n for n in layer['nodes']}
    reports = {n['id'] for n in layer['nodes'] if n['kind']=='report'}
    parents = [e for e in layer['edges'] if e['type']=='parent']
    with_parent = reports & {e['to'] for e in parents}
    resolved = reports & {e['to'] for e in parents if e['from'] in reports or e['from'] in roots}
    occ = layer['occurrences']
    out = {'nodes_by_kind': dict(collections.Counter(n['kind'] for n in layer['nodes'])),
        'nodes_by_status': dict(collections.Counter(n['status'] for n in layer['nodes'])),
        'edges_by_type': dict(collections.Counter(e['type'] for e in layer['edges'])),
        'parent_reports': len(with_parent), 'report_denominator': len(reports),
        'parent_report_fraction': len(with_parent)/len(reports), 'resolved_report_parent_count':len(resolved),
        'resolved_report_parent_fraction':len(resolved)/len(reports),
        'native_before': components(roots,old_pairs),
        'full_after': components(ids,old_pairs+[(e['from'],e['to']) for e in layer['edges']]),
        'parent_only_after': components(ids, old_pairs+[(e['from'],e['to']) for e in parents]),
        'indexed_occurrences_read':len(occ), 'actual_unique_indexed_hashes':len({o['actual_sha256'] for o in occ}),
        'exact_duplicate_occurrences':len(occ)-len({o['actual_sha256'] for o in occ}),
        'indexed_hash_mismatches':sum(not o['matches_index'] for o in occ),
        'unresolved_by_kind':dict(collections.Counter(u['kind'] for u in layer['unresolved'])),
        'source_id_conflicts':len(layer['id_conflicts']), 'cost':layer['cost']}
    return out


def export_cards(layer):
    cards=[]
    docs={d['path']:d for d in layer['documents']}
    program_doc=docs[str(PROGRAM)]
    program=json.loads(Path(program_doc['snapshot']).read_bytes())
    definitions=STORE.parent/'definitions'
    definitions.mkdir(exist_ok=True)
    if not (HERE/'DEFINITIONS').exists():(HERE/'DEFINITIONS').symlink_to(definitions, target_is_directory=True)
    layer_sha=sha((HERE/'BANK_LAYER.json').read_bytes())
    material=STORE.parent/'card_material'
    material.mkdir(exist_ok=True)
    if not (HERE/'CARD_MATERIAL').exists():(HERE/'CARD_MATERIAL').symlink_to(material, target_is_directory=True)
    incoming=collections.defaultdict(list)
    for e in layer['edges']:
        if e['type']=='parent':incoming[e['to']].append(e['from'])
    for n in layer['nodes']:
        if n['kind'] not in {'report','m_job'}:continue
        original=n.get('original_cards',[])
        c=dict(original[0]) if original else {}
        c.update({'id':n['id'], 'job':n['producer'], 'kind':n['kind'], 'title':n['title'],
            'check_level_claimed':c.get('check_level_claimed','unreviewed'),
            'result_summary':c.get('result_summary', n.get('excerpt', n.get('goal',''))),
            'obstruction_summary':c.get('obstruction_summary','UNKNOWN — inspect the full source; no inferred obstruction'),
            'Astra_reason':c.get('Astra_reason','historical producer record; independent review required'),
            'sources':[{'id':p,'family':'bank_parent'} for p in sorted(set(incoming[n['id']]))],
            'full_card':c.get('full_card',str(HERE/'BANK_LAYER.json')),
            'scientific_acceptance':False,
            'producer_report':next((a['path'] for a in n['aliases'] if isinstance(a,dict) and 'path'in a),str(PROGRAM)),
            'producer_report_sha256':n['sha256'], 'hash_kind':n['hash_kind'],
            'family':n['family'], 'source_family':n['source_family'], 'status':n['status'],
            'reviews':n['reviews'], 'parents':sorted(set(incoming[n['id']])),
            'aliases':n['aliases'], 'target':n['target'], 'time':n['time']})
        c['goal']=n.get('goal')
        if n['kind']=='m_job':
            c['program_row']=n['definition_evidence']
            definition=pointer(program,n['definition_evidence']['pointer'])
            path=definitions/(definition['id']+'.json')
            path.write_bytes(encoded(definition))
            c['producer_report']=str(path)
            c['hash_kind']='file_bytes_of_canonical_json_definition'
            c['full_card']=str(path)
            c['full_card_sha256']=n['sha256']
        else:
            c['original_producer_report']=c['producer_report']
            c['producer_report']=docs[c['producer_report']]['snapshot']
            path=material/(n['sha256']+'.json')
            raw=encoded(n)
            path.write_bytes(raw)
            c['original_full_card']=c['full_card']
            c['full_card']=str(path)
            c['full_card_sha256']=sha(raw)
        cards.append(c)
    return {'schema':'cards_all_bank_v1','cards':sorted(cards,key=lambda c:c['id']),
        'sampling_unit':'unique content report or versioned M definition; resource/reference nodes excluded',
        'unavailable_report_references_excluded':sum(n['kind']=='missing_report'for n in layer['nodes']),
        'bank_layer_sha256':layer_sha}


def sample_edges(layer):
    rng=random.Random(20261001)
    groups=collections.defaultdict(list)
    for e in layer['edges']:groups[e['type']].append(e)
    picked=[]
    while len(picked)<30:
        for typ in sorted(groups):
            remaining=[e for e in groups[typ] if e not in picked]
            if remaining and len(picked)<30:picked.append(rng.choice(remaining))
        if len(picked)==sum(map(len,groups.values())):break
    return {'selection':'seed 20261001, round-robin equal strata by edge type, then uniform within type',
        'sample':picked,'manual_verdicts_pending':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=int)
    args=p.parse_args()
    layer=Builder().build(args.pilot)
    if args.pilot:
        dump(HERE/'PILOT_METRICS.json',metrics(layer))
        print(json.dumps(metrics(layer),indent=1))
        return
    dump(HERE/'BANK_LAYER.json',layer)
    dump(HERE/'METRICS.json',metrics(layer))
    dump(HERE/'CARDS_ALL.json',export_cards(layer))
    dump(HERE/'EDGE_SAMPLE.json',sample_edges(layer))
    print(json.dumps(metrics(layer),indent=1))


if __name__=='__main__':main()
