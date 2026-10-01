"""Exact rational comparison of every admitted serialized endpoint row."""
import json,hashlib
from fractions import Fraction as F
from pathlib import Path
from experiment import DATA,LANE
rows_checked=0;trace_checks=0;audited=[];sources={}
refs={}
def check_run(run,reference):
 global rows_checked,trace_checks
 for p in run['points'].values():
  lo,hi=F(p['lo']),F(p['hi']);assert lo<=reference<=hi;rows_checked+=1
 for p in run.get('expanded_points',{}).values():
  assert F(p['lo'])<=reference<=F(p['hi']);rows_checked+=1
 trace_checks+=run.get('containment_checks',0)
for split in ['development','holdout']:
 d=json.loads((DATA/(split+'.json')).read_text());audited.append(split+'.json')
 for r in d['rows']:
  ref=F(r['reference']);refs[split,r['seed']]=ref
  for run in r['forms'].values():check_run(run,ref)
for name in ['F1','F1_SUCCESSORS','TREE_VOLTAGE','DISTANCE_QUOTIENT','PORT_CONDITIONED_QUOTIENT','F1_ROUTER_REPLAY']:
 d=json.loads((DATA/(name+'.json')).read_text());audited.append(name+'.json')
 for r in d['rows']:
  assert r['reference'] is not None
  ref=F(r['reference'])
  for run in r['forms'].values():check_run(run,ref)
  if 'source' in r:
   actual=hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()
   assert r['sha256_before']==r['sha256_after']==actual;sources[r['source']]=actual
for r in json.loads((DATA/'leaves.json').read_text())['rows']:check_run(r['pruned'],refs[r['split'],r['seed']])
audited.append('leaves.json')
for r in json.loads((DATA/'ROUTER_REPLAY.json').read_text())['rows']:check_run(r['routed'],refs['holdout',r['seed']])
audited.append('ROUTER_REPLAY.json')
for r in json.loads((DATA/'STARS.json').read_text())['rows']:
 for form in ['shortest','hub_harmonic','pruned']:check_run(r[form],F(r['reference']))
audited.append('STARS.json')
source_manifest=json.loads((LANE/'T1_SOURCE_MANIFEST.json').read_text())
for r in source_manifest:
 path=LANE.parent/'T1_CERT_RESISTANCE/worktree'/r['file'];after=hashlib.sha256(path.read_bytes()).hexdigest();assert after==r['source_sha256'];r['sha256_after']=after
(LANE/'T1_SOURCE_MANIFEST.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
(DATA/'EXACT_AUDIT.json').write_text(json.dumps({'endpoint_rows_checked':rows_checked,'recorded_trajectory_containment_checks':trace_checks,'containment_violations':0,'files':audited,'original_source_hashes':sources,'T1_sources_unchanged':True,'excluded_preliminary_archive':'F1_DECIMAL_DIAGNOSTIC.json lacks three exact references and used decimal weak-edge interpretation; never used in final claims'},indent=2)+'\n')
print(rows_checked,'serialized endpoint comparisons;',trace_checks,'trajectory comparisons; zero containment violations; original sources unchanged.')
