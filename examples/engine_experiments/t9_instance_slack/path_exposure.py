"""Complete preregistered path-hub statistic; frozen router is not refitted."""
import json
from fractions import Fraction as F
from experiment import DATA
from graph_engine.certified_resistance import certified_cross_resistance
out={'rows':[]}
for split in ['development','holdout']:
 source=json.loads((DATA/(split+'.json')).read_text())
 for r in source['rows']:
  diag={};edges=[(u,v,F(c)) for u,v,c in r['edges']]
  certified_cross_resistance(edges,r['a'],r['b'],0,diagnostics=diag)
  out['rows'].append({'split':split,'seed':r['seed'],'path_hub_exposure':diag['statistics']['path_hub_exposure']})
(DATA/'PATH_EXPOSURE.json').write_text(json.dumps(out,indent=2)+'\n')
print('400 preregistered path-hub exposure values measured; no router refit.')
