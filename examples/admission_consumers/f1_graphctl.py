"""Lane adapter: read native GRAPH.json and BANK_LAYER.json side by side.

Native validation, source health, refresh and ranking use the original graphctl.
Producer history projects to DEFERRED for the pinned engine. Documentary edges
remain separate from scientific depends_on. No source write or record is allowed.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from bank import HERE, PILOT, TOKEN, dump, pointer, sha


def native():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    spec = importlib.util.spec_from_file_location('native_graphctl', Path(__file__).resolve().parent / 'graphctl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aliases(n):
    return {a.get('id') for a in n.get('aliases',[]) if isinstance(a,dict)} | set(n.get('card_ids',[]))


def check_layer(layer, root_graph, verify_live=False):
    errors=[]
    docs={d['path']:d for d in layer['documents']}
    raw={}
    parsed={}
    def parse(h):
        if h not in parsed:parsed[h]=json.loads(raw[h])
        return parsed[h]
    live_drift=[]
    for path,d in docs.items():
        p=Path(d['snapshot'])
        if d['sha256'] not in raw:
            try:
                b=p.read_bytes()
                if sha(b)!=d['sha256']:raise ValueError('snapshot hash mismatch')
                raw[d['sha256']]=b
            except (OSError,ValueError) as e:
                errors.append({'type':'bad_snapshot','path':path,'error':str(e)})
        if verify_live:
            try:
                if sha(Path(path).read_bytes())!=d['sha256']:live_drift.append(path)
            except OSError:live_drift.append(path)
    nodes={n['id']:n for n in layer['nodes']}
    roots={n['id'] for n in root_graph['nodes']}
    ids=set(nodes)|roots
    if len(nodes)!=len(layer['nodes']) or set(nodes)&roots:errors.append({'type':'node_identity_collision'})
    for n in nodes.values():
        for k in ('producer','sha256','family','source_family','target','goal','time'):
            if k not in n:errors.append({'type':'missing_bank_field','node':n['id'],'field':k})
        if n.get('kind') not in {'report','m_job','missing_report','source_reference','resource'}:
            errors.append({'type':'unknown_bank_kind','node':n['id']})
        if n.get('status') not in {'PENDING','REVIEWED_SCOPED'} or n.get('scientific_acceptance') or n.get('root_verified'):
            errors.append({'type':'producer_admission','node':n['id']})
        if n['status']=='REVIEWED_SCOPED' and not n.get('reviews'):errors.append({'type':'review_status_without_review','node':n['id']})
        if n['kind']=='report':
            if n['id']!='B-'+n['sha256'] or n['sha256'] not in raw:errors.append({'type':'bad_report_identity','node':n['id']})
            for a in n['aliases']:
                if a.get('path') not in docs or docs[a['path']]['sha256']!=n['sha256']:errors.append({'type':'bad_alias','node':n['id']})
        if n['kind']=='m_job':
            w=n['definition_evidence']
            try:
                row=pointer(parse(w['sha256']),w['pointer'])
                from bank import encoded
                if sha(encoded(row))!=n['sha256']:raise ValueError('M row hash')
            except (KeyError,ValueError,IndexError,TypeError) as e:errors.append({'type':'bad_m_definition','node':n['id'],'error':str(e)})
        if n['kind']=='missing_report':
            w=n['definition_evidence']
            row=pointer(parse(w['sha256']),w['pointer'])
            if row.get('producer_report_sha256')!=n['sha256'] or row.get('job')!=n['producer'] or n.get('content_available') is not False:
                errors.append({'type':'bad_missing_report_binding','node':n['id']})
        for r in n.get('reviews',[]):
            if r['authority'] not in {'root_review','graphctl_x_ledger','COORDINATOR_REVIEWS'} or r.get('scientific_acceptance'):
                errors.append({'type':'bad_review_authority','node':n['id']})
            if r['document'] not in docs or docs[r['document']]['sha256']!=r['sha256']:
                errors.append({'type':'review_evidence_missing','node':n['id']})
    edges=layer['edges']
    if len({e['id'] for e in edges})!=len(edges):errors.append({'type':'duplicate_edge'})
    for e in edges:
        try:
            if e['from'] not in ids or e['to'] not in ids or e['from']==e['to']:raise ValueError('endpoint')
            if e.get('scientific_dependency') or not e['evidence']:raise ValueError('dependency or missing evidence')
            for w in e['evidence']:
                if w['document'] not in docs or docs[w['document']]['sha256']!=w['sha256']:raise ValueError('document binding')
                b=raw[w['sha256']]
                ref=w.get('reference')
                val=pointer(parse(w['sha256']),w['pointer']) if 'pointer'in w else None
                if 'line'in w:
                    line=b.decode('utf-8',errors='replace').splitlines()[w['line']-1]
                    if e['type']=='cited_id' and ref not in {m.group() for m in TOKEN.finditer(line)}:raise ValueError('citation token')
                if e['type']=='parent':
                    rid=val.get('id') if isinstance(val,dict) else val
                    if rid!=ref or ref not in aliases(nodes[e['from']]):raise ValueError('parent binding')
                    if w.get('declared_hash') and w['declared_hash']!=nodes[e['from']]['sha256']:raise ValueError('parent hash')
                elif e['type']=='cited_id':
                    if ref not in aliases(nodes[e['to']]) or nodes[e['from']]['sha256']!=w['sha256']:raise ValueError('citation endpoints')
                elif e['type']=='same_m_id':
                    if nodes[e['from']]['kind']!='m_job':raise ValueError('M source')
                    if ref!=nodes[e['from']]['job_id'] or ref not in aliases(nodes[e['to']]):raise ValueError('M identity')
                    if val!=ref and (not isinstance(val,dict) or val.get('job_id')!=ref):raise ValueError('M witness')
                elif e['type'] in {'combination_candidate','discriminating_test_candidate'}:
                    if 'new_card' in val:
                        if val['new_card']['id'] not in aliases(nodes[e['from']]) or val['old_card']['id'] not in aliases(nodes[e['to']]):raise ValueError('combination endpoints')
                    elif val.get('mutation_id') not in aliases(nodes[e['to']]):raise ValueError('bridge endpoint')
                    elif (val.get('relation')=='PROPOSED_DISCRIMINATING_TEST_CONSUMER')!=(e['type']=='discriminating_test_candidate'):
                        raise ValueError('test consumer is not method combination')
                elif e['type']=='shared_code_data':
                    if val!=ref:raise ValueError('resource name')
                    d=docs[w['resource_document']]
                    if d['sha256']!=w['resource_sha256'] or nodes[e['to']]['sha256']!=d['sha256']:raise ValueError('resource hash')
                    if not any(str(Path(a.get('path','')).parent/'inputs/SOURCE_CATALOG.json')==w['document'] for a in nodes[e['from']]['aliases']):raise ValueError('resource owner')
                    job=parse(docs[w['job_document']]['sha256'])
                    if w['source_key'] not in job.get('source_keys',[]):raise ValueError('source not selected')
                elif e['type']=='root_review_link':
                    row=json.loads(b.decode().splitlines()[w['line']-1])
                    if row['node']!=e['from'] or row['artifact']['sha256']!=w['review_sha256']:raise ValueError('root review fold')
                    if not any(r['document']==w['review_document'] for r in nodes[e['to']]['reviews']):raise ValueError('review target')
                else:raise ValueError('unknown edge type')
        except (KeyError,ValueError,IndexError,TypeError) as exc:
            errors.append({'type':'bad_edge','edge':e['id'],'reason':str(exc)})
    return {'ok':not errors,'errors':errors,'nodes':len(nodes),'edges':len(edges),
        'frozen_documents':len(docs),'live_drift':live_drift,'live_check':verify_live}


def load(layer_path):
    layer=json.loads(layer_path.read_text())
    doc=next(d for d in layer['documents'] if d['path']==layer['native_graph'])
    root=json.loads(Path(doc['snapshot']).read_bytes())
    ledger=next(d for d in layer['documents'] if d['path']==str(PILOT/'LEDGER.jsonl'))
    return layer,root,Path(ledger['snapshot'])


def compute(layer_path, verify_live=False):
    layer,root,ledger=load(layer_path)
    original=native()
    native_out=original.compute(root,ledger)
    check=check_layer(layer,root,verify_live)
    # Pinned engine has no PENDING status: DEFERRED is the closed projection.
    projected=copy.deepcopy(root)
    for n in layer['nodes']:
        projected['nodes'].append({'id':n['id'],'claim':n['title'],'type':'DATA-ANCHOR',
            'status':'DEFERRED','evidence':[],'depends_on':[],'load':0,'risk':'SOURCE-COND',
            'regime_note':'Producer documentary history only; independent claim review required.'})
    eng=original.load_tools().validate(projected,ledger_path=ledger,repo_root=PILOT)
    return layer,{'ok':native_out['validation']['ok'] and check['ok'] and eng['ok'],
        'bank_layer_sha256':sha(layer_path.read_bytes()),
        'native':native_out['validation'], 'layer':check, 'combined_engine':eng,
        'projection':'PENDING and REVIEWED_SCOPED → DEFERRED; provenance edges do not become proof dependencies'},native_out


def history(layer, nid):
    nodes={n['id']:n for n in layer['nodes']}
    if nid not in nodes:
        matches=[n for n in nodes.values() if nid in aliases(n)]
        if len(matches)!=1:raise ValueError(f'ID resolves to {len(matches)} nodes; use content ID')
        nid=matches[0]['id']
    related=[e for e in layer['edges'] if nid in {e['from'],e['to']}]
    resources={e['to'] for e in related if e['type']=='shared_code_data' and e['from']==nid}
    shared=[e for e in layer['edges'] if e['type']=='shared_code_data' and e['to'] in resources and e['from']!=nid]
    return {'node':nodes[nid],'edges':related,'shared_resource_neighbours':shared,
        'note':'parent/citation/resource sharing describes history; inspect reviews and assumptions before reuse'}


def admission_preview(layer, submissions, probe_template, base_dir, admission_guards=('a','b','c'),
                       verification_registry=None):
    """Read-only result proposals beside the bank; documentary links give no votes.

    Admission is scoped to these proposed receipts. It does not promote producer
    reports or rewrite the bank's PENDING/scientific_acceptance fields.
    """
    from graph_engine.tools.fold_gate_v2 import fold_gate_v2_round
    ids = {n['id'] for n in layer['nodes']}
    if any(s.get('node') not in ids for s in submissions):
        raise ValueError('submission must bind a bank content ID')
    return fold_gate_v2_round(submissions, probe_template, base_dir,
                              admission_guards=admission_guards, verification_registry=verification_registry)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['validate','rank','history'])
    p.add_argument('--layer',type=Path,default=HERE/'BANK_LAYER.json')
    p.add_argument('--node')
    p.add_argument('--check-live',action='store_true')
    args=p.parse_args()
    if args.command=='history':
        layer=json.loads(args.layer.read_text());out=history(layer,args.node)
        dump(HERE/'HISTORY_QUERY.json',out)
        print(json.dumps({'node':out['node']['id'],'status':out['node']['status'],'edges':len(out['edges']),
            'shared_resource_neighbours':len(out['shared_resource_neighbours']),'output':str(HERE/'HISTORY_QUERY.json')}))
        return
    layer,v,native_out=compute(args.layer,args.check_live)
    dump(HERE/'VALIDATION.json',v)
    if args.command in {'rank','validate'}:
        root_priority=native_out['priority'] if v['ok'] else []
        for n in root_priority:
            n['history_nodes']=[e['to'] for e in layer['edges'] if e['from']==n['id'] and e['type']=='root_review_link']
        dump(HERE/'PRIORITY.json',root_priority)
        dump(HERE/'NEXT_ACTIONS.json',native_out['next_actions'] if v['ok'] else [])
    print(json.dumps({'ok':v['ok'],'native_nodes':v['native']['node_count'],
        'combined_nodes':v['combined_engine']['node_count'],'layer_nodes':v['layer']['nodes'],
        'layer_edges':v['layer']['edges'],'errors':v['layer']['errors'][:10],
        'engine_errors':v['combined_engine']['errors'][:10],'native_errors':v['native']['errors'][:10],
        'native_warnings':v['native']['warning_count'],'live_drift':len(v['layer']['live_drift'])},indent=1))
    if not v['ok']:raise SystemExit(2)


if __name__=='__main__':main()
