"""Executed changed operations after small routing gain / F1 hub avoidance loss."""
import argparse,json,resource
from fractions import Fraction as F
from time import perf_counter,process_time
from collections import defaultdict
from experiment import DATA,measure,leaf_measure,BUDGETS
from f1_experiment import cases
from graph_engine.certified_resistance import ResistanceWitness,verify_resistance_witness

def stars():
 out={'rows':[]};start=perf_counter();cpu=process_time()
 for n in [32,256,4096]:
  e=[(0,u,F(1)) for u in range(1,n)];a,b=1,2
  # Independent full-edge Kirchhoff reference, not the candidate distance phi.
  p={u:F(1,2) for u in range(n)};p[a]=F(1);p[b]=F(0);curr=defaultdict(F)
  for u,v,c in e:cur=c*(p[u]-p[v]);curr[u]+=cur;curr[v]-=cur
  assert curr[a]==F(1,2) and curr[b]==-F(1,2) and all(x==0 for u,x in curr.items() if u not in [a,b])
  row={'family':'star','n':n,'a':a,'b':b,'edges':[[u,v,str(c)] for u,v,c in e],'reference':'2','reference_kind':'independent exact full-edge harmonic KCL'}
  row['shortest']=measure(e,a,b,F(2))
  row['hub_harmonic']=measure(e,a,b,F(2),initial_potential='hub_harmonic')
  row['pruned']=leaf_measure(row)
  out['rows'].append(row);print(n,row['shortest']['points']['0']['gap_lo'],row['pruned']['points']['0']['gap_lo'],flush=True)
 out.update(wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (DATA/'STARS.json').write_text(json.dumps(out,separators=(',',':'))+'\n')

def f1():
 refs={r['name']:r for r in json.loads((DATA/'F1.json').read_text())['rows']};out={'rows':[]};start=perf_counter();cpu=process_time()
 for name,e,a,b,path,sha in cases():
  ref=F(refs[name]['reference']);row={'name':name,'reference':str(ref),'forms':{}}
  for form in ['shortest','edge_disjoint']:
   row['forms']['cg_'+form]=measure(e,a,b,ref,form,12,[0,4,12],refinement='cg')
  row['forms']['hub_harmonic_cycle']=measure(e,a,b,ref,initial_potential='hub_harmonic')
  row['forms']['hub_harmonic_cg']=measure(e,a,b,ref,limit=12,checkpoints=[0,4,12],refinement='cg',initial_potential='hub_harmonic')
  # Original source is unchanged; extension is verified against original edges.
  spec={'edges':[[u,v,str(c)] for u,v,c in e],'a':a,'b':b,'reference':str(ref)}
  row['forms']['leaf_pruned']=leaf_measure(spec)
  out['rows'].append(row);print(name,{k:{b:round(v['gap_lo'],7) for b,v in r['points'].items()} for k,r in row['forms'].items()},flush=True)
  (DATA/'F1_SUCCESSORS.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
 out.update(wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (DATA/'F1_SUCCESSORS.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['stars','f1']);args=p.parse_args();stars() if args.mode=='stars' else f1()
