import json,math
from pathlib import Path
import analysis
from experiment import DATA
extra=json.loads((DATA/'PATH_EXPOSURE.json').read_text())['rows'];extra={(r['split'],r['seed']):r['path_hub_exposure'] for r in extra}
old=analysis.load
def augmented(split):
 d=old(split)
 for r in d['rows']:r['extra_statistics']['path_hub_exposure']=extra[(split,r['seed'])]
 return d
analysis.load=augmented;analysis.evaluate()
def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
p=DATA/'ANALYSIS.json';p.write_text(json.dumps(clean(json.loads(p.read_text())),indent=2,allow_nan=False)+'\n')
print('Completed all preregistered statistics; undefined correlations serialized as null; frozen router unchanged.')
