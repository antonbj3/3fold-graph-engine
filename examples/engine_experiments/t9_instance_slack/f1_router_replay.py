import json,resource,hashlib
from fractions import Fraction as F
from time import perf_counter,process_time
from experiment import DATA,measure
from f1_experiment import cases
start=perf_counter();cpu=process_time();refs={r['name']:r for r in json.loads((DATA/'F1.json').read_text())['rows']};cg={r['name']:r for r in json.loads((DATA/'F1_SUCCESSORS.json').read_text())['rows']};out={'rows':[]}
for name,e,a,b,path,sha in cases():
 ref=F(refs[name]['reference']);row={'name':name,'reference':str(ref),'forms':{},'source':str(path),'sha256_before':sha}
 for mode,limit,budgets in [('cycle',512,[0,32,128,512]),('cg',12,[0,4,12])]:
  run=measure(e,a,b,ref,form='routed',limit=limit,checkpoints=budgets,refinement=mode);chosen=run['route_flow_selected']
  baseline=refs[name]['forms'][chosen] if mode=='cycle' else cg[name]['forms']['cg_'+chosen]
  for budget in budgets:
   for key in ['lo','hi','updates']:assert run['points'][str(budget)][key]==baseline['points'][str(budget)][key]
  assert run['first_sign']==baseline['first_sign'];row['forms'][mode]=run
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;row['sha256_after']=sha;out['rows'].append(row)
 print(name,row['forms']['cycle']['route_flow_selected'],flush=True)
 (DATA/'F1_ROUTER_REPLAY.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
out.update(wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(DATA/'F1_ROUTER_REPLAY.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
