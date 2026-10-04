"""T9 adversarial experiment: exact references, matched attempts, frozen routing."""
import argparse,hashlib,importlib.util,json,math,random,resource,sys,os
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
LANE=Path(os.environ.get("T9_LANE",Path(__file__).resolve().parent))
DATA=Path(os.environ.get('T9_DATA_DIR','t9_data'))  # set T9_DATA_DIR to a scratch disk; the default is repo-local;DATA.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(LANE/'worktree/src'));sys.set_int_max_str_digits(0)
from graph_engine.certified_resistance import ResistanceBudget,ResistanceWitness,certified_cross_resistance,verify_resistance_witness
from graph_engine.instance_slack import prune_leaves
spec=importlib.util.spec_from_file_location('oracle',LANE/'worktree/tests/test_certified_resistance.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
FAMILIES=['path','cycle','grid','star','hub_ring','preferential','random_sparse','dense','barbell','parallel']
BUDGETS=[0,32,128,512];RATIOS=[F(4,5),F(19,20),F(21,20),F(6,5)]

def graph(family,seed):
    rng=random.Random(seed);n=rng.randint(12,36);pairs=[];weights=False
    if family=='path':pairs=[(i,i+1) for i in range(n-1)];weights=True
    elif family=='cycle':pairs=[(i,(i+1)%n) for i in range(n)];weights=True
    elif family=='grid':
        side=rng.randint(4,6);n=side*side
        pairs=[(i,j) for i in range(n) for j in (i+1,i+side) if j<n and (j!=i+1 or i//side==j//side)]
    elif family=='star':pairs=[(0,i) for i in range(1,n)]
    elif family=='hub_ring':pairs=[(0,i) for i in range(1,n)]+[(i,1+i%(n-1)) for i in range(1,n)]
    elif family=='preferential':pairs=list(nx.barabasi_albert_graph(n,rng.choice([1,2,3]),seed=seed).edges());weights=True
    elif family=='random_sparse':pairs=[(i,rng.randrange(i)) for i in range(1,n)]+[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.06];weights=True
    elif family=='dense':pairs=[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.6]+[(i,i+1) for i in range(n-1)];weights=True
    elif family=='barbell':
        half=n//2;pairs=[(i,j) for i in range(n) for j in range(i+1,n) if (i<half)==(j<half) and (j==i+1 or rng.random()<.4)]+[(half-1,half)]
    elif family=='parallel':
        k=rng.randint(2,8);pairs=[(0,i) for i in range(2,k+2)]+[(i,1) for i in range(2,k+2)]+[(0,1)]
        pairs += [(rng.randrange(i),i) for i in range(k+2,n)];weights=True
    edges=[(u,v,rng.choice([F(1,4),F(1),F(2),F(5)]) if weights else F(1)) for u,v in pairs]
    if family=='barbell':edges[-1]=(edges[-1][0],edges[-1][1],F(1,rng.choice([10,100,1000])))
    if family=='star':a,b=rng.sample(range(1,n),2)
    elif family=='parallel':a,b=0,1
    else:a,b=rng.sample(range(n),2)
    # Permute all node labels and edge order before running. This challenges
    # sorted-coordinate and fundamental-cycle-order confounding.
    perm=list(range(n));rng.shuffle(perm);edges=[(perm[u],perm[v],c) for u,v,c in edges];rng.shuffle(edges)
    return edges,perm[a],perm[b]


def measure(edges,a,b,exact,form='shortest',limit=512,checkpoints=BUDGETS,refinement='cycle',initial_potential='distance'):
    started=perf_counter();cpu=process_time();diag={};points={};first={};checks=0;verification=0.
    def callback(lo,hi,k,flow,phi):
        nonlocal checks,verification
        if exact is not None:
            assert lo<=exact<=hi,(form,k,lo,exact,hi)
            checks+=1
        hits=[str(r) for r in RATIOS if exact is not None and str(r) not in first and (lo>r*exact or hi<r*exact)]
        if k in checkpoints or hits:
            w=ResistanceWitness('connected',tuple(flow),tuple(sorted(phi.items())),updates=k,input_validated=True)
            t=perf_counter();verify_resistance_witness(edges,a,b,lo,hi,w);verification+=perf_counter()-t
            row={'updates':k,'lo':str(lo),'hi':str(hi),'gap_lo':float((hi-lo)/lo),'gap_exact':float((hi-lo)/exact) if exact else None,
                 'elapsed_seconds':perf_counter()-started,'verification_seconds':verification}
            if k in checkpoints:points[str(k)]=row
            for r in hits:first[r]=k
    lo,hi,w=certified_cross_resistance(edges,a,b,ResistanceBudget(max_updates=limit,refinement=refinement),initial_flow=form,checkpoint=callback,diagnostics=diag,initial_potential=initial_potential)
    t=perf_counter();verify_resistance_witness(edges,a,b,lo,hi,w);verification+=perf_counter()-t
    if exact is not None:assert lo<=exact<=hi
    end={'updates':w.updates,'lo':str(lo),'hi':str(hi),'gap_lo':float((hi-lo)/lo),'gap_exact':float((hi-lo)/exact) if exact else None,
         'elapsed_seconds':perf_counter()-started,'verification_seconds':verification}
    for k in checkpoints:
        if k<=limit and str(k) not in points:points[str(k)]=dict(end)
    return {'form':form,'refinement':refinement,'initial_potential':initial_potential,'points':points,'first_sign':{str(r):first.get(str(r)) for r in RATIOS},
            'updates_actual':w.updates,'startup_seconds':w.startup_seconds,'wall_seconds':perf_counter()-started,'cpu_seconds':process_time()-cpu,
            'verification_seconds':verification,'containment_checks':checks+int(exact is not None),**diag}


def extra_stats(edges,a,b):
    t=perf_counter();g=nx.Graph();g.add_weighted_edges_from((u,v,float(c)) for u,v,c in edges)
    # Parallel rows in original certificate: Menger on multigraph can differ.
    # Compute original-edge capacities, not collapsed simple connectivity.
    capacity=nx.DiGraph()
    for u,v,c in edges:
        if u==v:continue
        for x,y in [(u,v),(v,u)]:
            if capacity.has_edge(x,y):capacity[x][y]['capacity']+=1
            else:capacity.add_edge(x,y,capacity=1)
    menger=nx.maximum_flow_value(capacity,a,b);tm=perf_counter()-t
    t=perf_counter();nodes=sorted(g);idx={u:i for i,u in enumerate(nodes)};lap=np.zeros((len(nodes),len(nodes)))
    for u,v,c in edges:
        if u==v:continue
        i,j=idx[u],idx[v];q=float(c);lap[i,i]+=q;lap[j,j]+=q;lap[i,j]-=q;lap[j,i]-=q
    degree=np.diag(lap);norm=lap/np.sqrt(degree[:,None]*degree[None,:]);spectral=float(np.linalg.eigvalsh(norm)[1]);ts=perf_counter()-t
    return {'menger_edge_count':menger,'normalized_spectral_gap':spectral}, {'menger_seconds':tm,'spectral_seconds':ts}


def run(split,count,output,forms):
    started=perf_counter();cpu=process_time();out={'split':split,'rows':[],'failures':0,'budgets':BUDGETS,'threshold_ratios':[str(r) for r in RATIOS]}
    for family in FAMILIES:
        for i in range(count):
            seed=(202610010900 if split=='development' else 202611091700)+1000*FAMILIES.index(family)+i
            edges,a,b=graph(family,seed);t=perf_counter();exact=oracle.independent_oracle(edges,a,b);ref=perf_counter()-t
            assert exact>0 and not math.isinf(exact)
            extras,tcost=extra_stats(edges,a,b)
            row={'family':family,'seed':seed,'a':a,'b':b,'edges':[[u,v,str(c)] for u,v,c in edges],'reference':str(exact),'reference_seconds':ref,'extra_statistics':extras,'extra_statistics_cost':tcost,'forms':{}}
            for form in forms:row['forms'][form]=measure(edges,a,b,exact,form)
            out['rows'].append(row)
        Path(output).write_text(json.dumps(out,separators=(',',':'))+'\n')
        print(json.dumps({'family':family,'rows':len(out['rows']),'wall':perf_counter()-started}),flush=True)
    out.update(wall_seconds=perf_counter()-started,cpu_seconds=process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    Path(output).write_text(json.dumps(out,separators=(',',':'))+'\n')
    return out


def leaf_measure(row,form='shortest'):
    edges=[(u,v,F(c)) for u,v,c in row['edges']];a,b=row['a'],row['b'];exact=F(row['reference'])
    t=perf_counter();reduced,kept,forest=prune_leaves(edges,a,b);prep=perf_counter()-t
    r=measure(reduced,a,b,exact,form)
    # Check all expanded checkpoint potentials and zero-current removed edges.
    expanded_checks=0;expansion_started=perf_counter();expanded_points={}
    for budget in BUDGETS:
        replay_start=perf_counter()
        lo,hi,w=certified_cross_resistance(reduced,a,b,ResistanceBudget(max_updates=budget),initial_flow=form)
        phi=dict(w.potential)
        for u,p in reversed(forest):phi[u]=phi[p] if p is not None else F(0)
        flow=[F(0)]*len(edges)
        for i,f in zip(kept,w.flow):flow[i]=f
        full=ResistanceWitness('connected',tuple(flow),tuple(sorted(phi.items())),updates=w.updates,input_validated=True)
        verify_resistance_witness(edges,a,b,lo,hi,full);assert lo<=exact<=hi;expanded_checks+=1
        expanded_points[str(budget)]={'total_seconds':perf_counter()-replay_start+prep,'lo':str(lo),'hi':str(hi),'updates':w.updates}
    r.update(pruning_seconds=prep,original_edges=len(edges),reduced_edges=len(reduced),pruned_nodes=len(forest),expanded_checks=expanded_checks,expanded_points=expanded_points,expansion_replay_seconds=perf_counter()-expansion_started)
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['pilot','development','holdout','leaves']);p.add_argument('--count',type=int,default=20);p.add_argument('--output');args=p.parse_args()
    if args.split=='leaves':
        t=perf_counter();c=process_time();o={'rows':[]}
        for split in ['development','holdout']:
            src=json.loads((DATA/(split+'.json')).read_text())
            for row in src['rows']:
                o['rows'].append({'split':split,'family':row['family'],'seed':row['seed'],'pruned':leaf_measure(row)})
            print(split,flush=True)
        o.update(wall_seconds=perf_counter()-t,cpu_seconds=process_time()-c,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (DATA/'leaves.json').write_text(json.dumps(o,separators=(',',':'))+'\n')
    else:run('development' if args.split=='pilot' else args.split,args.count,args.output or DATA/(args.split+'.json'),['shortest','edge_disjoint','retained_sp'] if args.split=='pilot' else ['shortest','edge_disjoint'])
