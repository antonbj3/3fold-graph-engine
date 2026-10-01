import json,resource,hashlib
from fractions import Fraction as F
from time import perf_counter,process_time
from experiment import DATA,measure
from f1_experiment import cases
start=perf_counter();cpu=process_time();refs={r['name']:r for r in json.loads((DATA/'F1.json').read_text())['rows']};out={'rows':[]}
for name,e,a,b,path,sha in cases():
 ref=F(refs[name]['reference']);row={'name':name,'source':str(path),'sha256_before':sha,'reference':str(ref),'forms':{}}
 row['forms']['tree_voltage_cycle']=measure(e,a,b,ref,initial_potential='tree_voltage')
 row['forms']['tree_voltage_cg']=measure(e,a,b,ref,limit=12,checkpoints=[0,4,12],refinement='cg',initial_potential='tree_voltage')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;row['sha256_after']=sha
 out['rows'].append(row)
 (DATA/'TREE_VOLTAGE.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
 print(name,{k:{b:round(p['gap_lo'],8) for b,p in f['points'].items()} for k,f in row['forms'].items()},flush=True)
out.update(wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(DATA/'TREE_VOLTAGE.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
