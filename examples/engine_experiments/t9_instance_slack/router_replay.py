import json,resource
from fractions import Fraction as F
from time import perf_counter,process_time
from experiment import DATA,measure,BUDGETS
start=perf_counter();cpu=process_time();source=json.loads((DATA/'holdout.json').read_text());router=json.loads((DATA/'FROZEN_ROUTER.json').read_text());out={'rows':[],'frozen_router':router}
for row in source['rows']:
 edges=[(u,v,F(c)) for u,v,c in row['edges']]
 run=measure(edges,row['a'],row['b'],F(row['reference']),form='routed')
 chosen=router['below'] if row['forms']['shortest']['statistics'][router['statistic']]<=router['threshold'] else router['above']
 assert run['route_flow_selected']==chosen
 for b in BUDGETS:
  for k in ['lo','hi','updates']:assert run['points'][str(b)][k]==row['forms'][chosen]['points'][str(b)][k]
 assert run['first_sign']==row['forms'][chosen]['first_sign']
 out['rows'].append({'seed':row['seed'],'family':row['family'],'routed':run,'matches_offline_selected_form_exactly':True})
out.update(wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(DATA/'ROUTER_REPLAY.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print('200 native router replays exactly match frozen-policy endpoints, update counts, and first-sign counts.')
