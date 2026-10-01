import json,resource,hashlib,sys
from fractions import Fraction as F
from time import perf_counter,process_time
from experiment import DATA,measure
from f1_experiment import cases
conditioned='--conditioned' in sys.argv;potential='port_conditioned_quotient' if conditioned else 'distance_quotient';artifact='PORT_CONDITIONED_QUOTIENT.json' if conditioned else 'DISTANCE_QUOTIENT.json'
start=perf_counter();cpu=process_time();refs={r['name']:r for r in json.loads((DATA/'F1.json').read_text())['rows']};out={'rows':[]}
for name,e,a,b,path,sha in cases():
 ref=F(refs[name]['reference']);row={'name':name,'source':str(path),'sha256_before':sha,'reference':str(ref),'forms':{}}
 for form in ['shortest','edge_disjoint']:
  row['forms']['quotient_cycle_'+form]=measure(e,a,b,ref,form,initial_potential=potential)
  row['forms']['quotient_cg_'+form]=measure(e,a,b,ref,form,limit=12,checkpoints=[0,4,12],refinement='cg',initial_potential=potential)
  initial=row['forms']['quotient_cycle_'+form]['points']['0'];baseline=refs[name]['forms'][form]['points']['0']
  assert F(initial['hi'])==F(baseline['hi']) and F(initial['lo'])>=F(baseline['lo'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;row['sha256_after']=sha
 out['rows'].append(row)
 (DATA/artifact).write_text(json.dumps(out,separators=(',',':'))+'\n')
 print(name,{k:{b:round(p['gap_lo'],8) for b,p in f['points'].items()} for k,f in row['forms'].items()},flush=True)
out.update(wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(DATA/artifact).write_text(json.dumps(out,separators=(',',':'))+'\n')
