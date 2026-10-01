"""Read-only F1 cases, exact sparse independent reference, same-attempt variants."""
import argparse,hashlib,json,heapq,resource
from collections import defaultdict,deque
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter,process_time
from experiment import LANE,DATA,measure
import networkx as nx
from graph_engine.instance_slack import prune_leaves
from graph_engine.certified_resistance import ResistanceBudget,ResistanceWitness,certified_cross_resistance,verify_resistance_witness

def far(adj,root):
 d={root:0};q=deque([root])
 while q:
  u=q.popleft()
  for v in sorted(adj[u]):
   if v not in d:d[v]=d[u]+1;q.append(v)
 return max(d,key=lambda u:(d[u],-u))

def cases():
 bank=LANE.parent/'F1_FOLD_BANK/BANK_LAYER.json';raw=bank.read_bytes();d=json.loads(raw);ids={v['id']:i for i,v in enumerate(d['nodes'])};edges=[(ids[x['from']],ids[x['to']],F(1)) for x in d['edges']]
 g=nx.Graph();g.add_nodes_from(range(len(ids)));g.add_edges_from((u,v) for u,v,c in edges)
 part=sorted(nx.connected_components(g),key=lambda x:(-len(x),min(x)))[0];a=far(g.adj,min(part));b=far(g.adj,a)
 yield 'F1_FULL_4679',edges,a,b,bank,hashlib.sha256(raw).hexdigest()
 for name in ['F1_FULL_3459','F1_3459_UNGROUNDED']:
  path=LANE.parent/'J3_LARGE_INTERFACE/cases'/(name+'.json');raw=path.read_bytes();d=json.loads(raw);shared=d['shared'];interior=sorted(set(range(d['n']))-set(shared));boundary={u:i for i,u in enumerate(shared)}
  aa={**boundary,**{u:len(shared)+i for i,u in enumerate(interior)}};bb={**boundary,**{u:len(shared)+len(interior)+i for i,u in enumerate(interior)}}
  rows=[]
  for mapping in [aa,bb]:rows += [(mapping[u],mapping[v],F(c)) for (u,v),c in zip(d['edges'],d['weights'])]
  for k in ([0] if name=='F1_FULL_3459' else [0,1]):
   x,y=d['queries'][k];yield name+'_Q'+str(k),rows,aa[interior[x]],bb[interior[y]],path,hashlib.sha256(raw).hexdigest()

def inspect():
 out=[]
 for name,e,a,b,p,h in cases():
  g=nx.Graph();g.add_edges_from((u,v) for u,v,c in e if u!=v)
  blocks=list(nx.biconnected_components(g));reduced,kept,forest=prune_leaves(e,a,b)
  deg=sorted(dict(g.degree).values());item={'name':name,'n':len(g),'m_rows':len(e),'a':a,'b':b,'leaf_pruned_nodes':len(forest),'reduced_edges':len(reduced),'largest_biconnected':max(map(len,blocks)),'blocks':len(blocks),'source':str(p),'sha256':h}
  out.append(item);print(json.dumps(item),flush=True)
 (DATA/'F1_INSPECT.json').write_text(json.dumps(out,indent=2)+'\n')


def exact_sparse(edges,a,b,max_seconds=120,backend="flint"):
 """Exact star-mesh elimination plus independent full-edge KCL residual check."""
 start=perf_counter();g=defaultdict(dict)
 if backend=="flint":
  import sys
  sys.path.insert(0,str(DATA/'vendor'))
  import flint
  flint.ctx.threads=2
  def Q(v):
   x=F(v);return flint.fmpq(x.numerator,x.denominator)
 else:Q=F
 for u,v,c in edges:
  if u==v:continue
  c=Q(c);g[u][v]=g[u].get(v,Q(0))+c;g[v][u]=g[v].get(u,Q(0))+c
 seen={b};stack=[b]
 while stack:
  for v in g[stack.pop()]:
   if v not in seen:seen.add(v);stack.append(v)
 if a not in seen:return None,{'disconnected':True}
 g={u:{v:c for v,c in ns.items() if v in seen} for u,ns in g.items() if u in seen}
 heap=[(len(ns),u) for u,ns in g.items() if u not in (a,b)];heapq.heapify(heap);forest=[];updates=0;max_degree=0
 while heap:
  degree,u=heapq.heappop(heap)
  if u not in g:continue
  if degree!=len(g[u]):heapq.heappush(heap,(len(g[u]),u));continue
  if perf_counter()-start>max_seconds:raise TimeoutError((len(g),max_degree))
  ns=list(g[u].items());diagonal=sum(c for v,c in ns);forest.append((u,diagonal,ns));max_degree=max(max_degree,len(ns))
  for i,(v,cv) in enumerate(ns):
   del g[v][u]
   for w,cw in ns[i+1:]:
    value=g[v].get(w,Q(0))+cv*cw/diagonal;g[v][w]=value;g[w][v]=value
   if v not in (a,b):heapq.heappush(heap,(len(g[v]),v))
  del g[u];updates+=1
 conductance=g[a][b];potential={a:Q(1),b:Q(0)}
 for u,d,ns in reversed(forest):potential[u]=sum(c*potential[v] for v,c in ns)/d
 residual=defaultdict(lambda:Q(0))
 for u,v,c in edges:
  if u not in seen:continue
  current=Q(c)*(potential[u]-potential[v]);residual[u]+=current;residual[v]-=current
 assert residual[a]==conductance and residual[b]==-conductance
 assert all(v==0 for u,v in residual.items() if u not in (a,b))
 answer=1/conductance
 answer=F(int(answer.numerator),int(answer.denominator))
 return answer,{'backend':backend,'wall_seconds':perf_counter()-start,'eliminations':updates,'max_elimination_degree':max_degree,'exact_full_edge_residual':True}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['inspect','exact_pilot','run']);args=p.parse_args()
 if args.mode=='inspect':inspect()
 elif args.mode=='exact_pilot':
  name,e,a,b,_,_=next(cases());ref,receipt=exact_sparse(e,a,b,30);print(name,float(ref),receipt)
 else:
  out={'rows':[]};t=perf_counter();cpu=process_time()
  for name,e,a,b,path,sha in cases():
   try:ref,receipt=exact_sparse(e,a,b,420)
   except TimeoutError as exc:ref=None;receipt={'timeout':repr(exc),'reference_unavailable':True}
   row={'name':name,'a':a,'b':b,'source':str(path),'sha256_before':sha,'reference':str(ref) if ref is not None else None,'reference_receipt':receipt,'forms':{}}
   for form in ['shortest','edge_disjoint']:
    row['forms'][form]=measure(e,a,b,ref,form,512)
    print(name,form,row['forms'][form]['points']['128']['gap_lo'],flush=True)
   assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;row['sha256_after']=sha;out['rows'].append(row)
   (DATA/'F1.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
  out.update(wall_seconds=perf_counter()-t,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  (DATA/'F1.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
