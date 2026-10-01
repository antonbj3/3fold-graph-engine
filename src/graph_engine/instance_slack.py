"""Cheap structural statistics and bounded alternative feasible flows for T9."""
from fractions import Fraction as F
import heapq
from collections import deque


def avoided_hub_flow(rows, adj, a, b, check=lambda: None, cap=8):
    """Greedy edge-disjoint paths avoiding the largest nonterminal degree hub.

    No exact Menger claim. Missing avoided-hub support returns None (SP fallback).
    Inverse path energy weights are the exact minimum-energy current split on
    this edge-disjoint path support. Construction work is separate from updates.
    """
    candidates=[u for u in adj if u not in (a,b)]
    hub=max(candidates,key=lambda u:(len(adj[u]),-u)) if candidates else None
    used=set(); paths=[]
    for _ in range(cap):
        dist={a:F(0)};parent={};heap=[(F(0),a)];done=set()
        while heap:
            check();d,u=heapq.heappop(heap)
            if u in done: continue
            done.add(u)
            if u==b:break
            for v,i,sign in adj[u]:
                check()
                if i in used or v==hub:continue
                nd=d+1/rows[i][2]
                if v not in dist or nd<dist[v]:
                    dist[v]=nd;parent[v]=(u,i,sign);heapq.heappush(heap,(nd,v))
        if b not in done:break
        path=[];u=b
        while u!=a:
            v,i,sign=parent[u];path.append((i,sign));used.add(i);u=v
        paths.append((dist[b],path))
    if not paths:return None,{'hub':hub,'paths':0,'fallback':True}
    conductance=sum(1/r for r,_ in paths);flow=[F(0)]*len(rows)
    for r,path in paths:
        weight=1/r/conductance
        for i,sign in path:flow[i]=sign*weight
    return flow,{'hub':hub,'paths':len(paths),'fallback':False}


def structural_stats(rows, adj, a, b, dist, initial_lo, initial_hi, parent=None):
    nodes=list(dist);degrees=sorted(len(adj[u]) for u in nodes);n=len(nodes)
    m=sum(degrees)/2;total=sum(degrees)
    wd={u:sum((rows[i][2] for _,i,_ in adj[u]),F(0)) for u in nodes}
    gini=sum((2*(i+1)-n-1)*d for i,d in enumerate(degrees))/(n*total) if total else 0
    # Equivalent to a leaf-peeling traversal; endpoints protected. This is a
    # separate O(m+n) predictor and its cost is explicitly measured.
    degree={u:len(adj[u]) for u in nodes};q=deque(u for u in nodes if degree[u]<=1 and u not in (a,b));removed=set()
    while q:
        u=q.popleft()
        if u in removed:continue
        removed.add(u)
        for v,_,_ in adj[u]:
            if v not in removed:
                degree[v]-=1
                if degree[v]<=1 and v not in (a,b):q.append(v)
    path_hub=0;u=a
    if parent is not None:
        while u!=b:
            if u!=a:path_hub=max(path_hub,len(adj[u]))
            u=parent[u][0]
    return {'path_hub_exposure':path_hub/m,'n':n,'m':m,'max_degree_over_mean':max(degrees)*n/total,
            'degree_gini':gini,'hub_edge_share':max(degrees)/m,
            'port_min_degree':min(len(adj[a]),len(adj[b])),
            'port_inverse_weighted_degree':float(1/wd[a]+1/wd[b]),
            'shortest_resistance':float(dist[a]),
            'distance_over_sink_eccentricity':float(dist[a]/max(dist.values())),
            'leaf_prunable_fraction':len(removed)/n,
            'initial_certificate_gap':float((initial_hi-initial_lo)/initial_lo)}


def prune_leaves(edges,a,b):
    """Exact two-terminal leaf support; return parent forest for extension."""
    adj={a:[],b:[]}
    for i,(u,v,c) in enumerate(edges):
        adj.setdefault(u,[]);adj.setdefault(v,[])
        if u!=v:adj[u].append((v,i));adj[v].append((u,i))
    degree={u:len(vs) for u,vs in adj.items()};removed=set();forest=[]
    queue=deque(u for u,d in degree.items() if d<=1 and u not in (a,b))
    for_loop=0
    while queue:
        u=queue.popleft()
        if u in removed:continue
        live=[(v,i) for v,i in adj[u] if v not in removed]
        if live:forest.append((u,live[0][0]))
        else:forest.append((u,None))
        removed.add(u)
        for v,i in live:
            degree[v]-=1
            if degree[v]<=1 and v not in (a,b):queue.append(v)
    kept=[i for i,(u,v,_) in enumerate(edges) if u not in removed and v not in removed]
    return [edges[i] for i in kept],kept,forest


def select_initial_flow(adj,component,threshold=3.245):
    """Frozen development policy; uses only already built incidence degrees."""
    degrees=[len(adj[u]) for u in component]
    statistic=max(degrees)*len(degrees)/sum(degrees)
    return ("edge_disjoint" if statistic<=threshold else "shortest"),statistic


def distance_quotient_potential(rows,adj,a,b,dist,check=lambda:None,cap=64,condition_ports=False):
    """Exact Dirichlet optimum on already known sink-distance classes.

    A reduced solve, never a full original-vertex Laplacian. Original distance
    potential belongs to the subspace, so this lower bound cannot worsen at start.
    """
    keys={u: (0,u) if condition_ports and u in (a,b) else (1,d) for u,d in dist.items()}
    levels=sorted(set(keys.values()))
    if len(levels)>cap:return None,{'quotient_groups':len(levels),'quotient_fallback':True}
    groups={d:i for i,d in enumerate(levels)};ga,gb=groups[keys[a]],groups[keys[b]]
    unknown=[i for i in range(len(levels)) if i not in (ga,gb)];index={u:i for i,u in enumerate(unknown)}
    matrix=[[F(0)]*(len(unknown)+1) for _ in unknown]
    for u,v,c in rows:
        check()
        if u not in dist or u==v:continue
        x,y=groups[keys[u]],groups[keys[v]]
        if x==y:continue
        for p,q in [(x,y),(y,x)]:
            if p in index:
                i=index[p];matrix[i][i]+=c
                if q in index:matrix[i][index[q]]-=c
                elif q==ga:matrix[i][-1]+=c
    operations=0;n=len(unknown)
    for k in range(n):
        check();assert matrix[k][k]>0
        for i in range(k+1,n):
            if matrix[i][k]:
                factor=matrix[i][k]/matrix[k][k]
                for j in range(k,n+1):
                    matrix[i][j]-=factor*matrix[k][j];operations+=1
    solution=[F(0)]*n
    for i in reversed(range(n)):
        check();solution[i]=(matrix[i][-1]-sum(matrix[i][j]*solution[j] for j in range(i+1,n)))/matrix[i][i]
    gp={ga:F(1),gb:F(0),**{g:solution[index[g]] for g in unknown}}
    phi={u:gp[groups[keys[u]]] if u in dist else F(0) for u in adj}
    return phi,{'quotient_groups':len(levels),'quotient_unknowns':n,'quotient_elimination_updates':operations,'quotient_fallback':False,'ports_conditioned':condition_ports}
