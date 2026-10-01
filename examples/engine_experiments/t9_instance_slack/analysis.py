import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from experiment import DATA,RATIOS,FAMILIES

def load(split):return json.loads((DATA/(split+'.json')).read_text())
def feature(row,key):return row['forms']['shortest']['statistics'].get(key,row['extra_statistics'].get(key))
def gap(row,form,budget=128,kind='gap_lo'):return row['forms'][form]['points'][str(budget)][kind]
def loss(values):return float(np.mean(np.log1p(values)))
def stats_keys(rows):return list(rows[0]['forms']['shortest']['statistics'])+list(rows[0]['extra_statistics'])

def correlations(rows):
 out={}
 for key in stats_keys(rows):
  x=[feature(r,key) for r in rows];record={}
  for budget in [0,32,128,512]:
   record[str(budget)]={k:float(spearmanr(x,[gap(r,'shortest',budget,k) for r in rows]).statistic) for k in ['gap_lo','gap_exact']}
  record['first_sign']={str(q):{'rho_censored_ordinal':float(spearmanr(x,[r['forms']['shortest']['first_sign'][str(q)] if r['forms']['shortest']['first_sign'][str(q)] is not None else 513 for r in rows]).statistic),
   'censored':sum(r['forms']['shortest']['first_sign'][str(q)] is None for r in rows)} for q in RATIOS}
  record['within_family_128']={f:float(spearmanr([feature(r,key) for r in rows if r['family']==f],[gap(r,'shortest') for r in rows if r['family']==f]).statistic) for f in FAMILIES}
  out[key]=record
 return out

def train():
 rows=load('development')['rows'];keys=[k for k in stats_keys(rows) if k not in ['initial_certificate_gap','menger_edge_count','normalized_spectral_gap']]
 best=None
 for key in keys:
  vals=sorted(set(feature(r,key) for r in rows));cuts=[-math.inf]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[math.inf]
  for cut in cuts:
   for below in ['shortest','edge_disjoint']:
    above='edge_disjoint' if below=='shortest' else 'shortest'
    forms=[below if feature(r,key)<=cut else above for r in rows]
    objective=loss([gap(r,f) for r,f in zip(rows,forms)])
    if best is None or objective<best['development_objective']:
     best={'statistic':key,'threshold':cut,'below':below,'above':above,'development_objective':objective,'budget':128,'selection':'minimum development mean log1p(gap_lo), single threshold'}
 (DATA/'FROZEN_ROUTER.json').write_text(json.dumps(best,indent=2)+'\n')
 (DATA/'development_analysis.json').write_text(json.dumps({'correlations':correlations(rows),'router':best},indent=2)+'\n')
 print(json.dumps(best))

def evaluate():
 router=json.loads((DATA/'FROZEN_ROUTER.json').read_text());o={'router':router,'splits':{}}
 for split in ['development','holdout']:
  data=load(split);rows=data['rows'];rforms=[router['below'] if feature(r,router['statistic'])<=router['threshold'] else router['above'] for r in rows]
  result={'n':len(rows),'correlations':correlations(rows),'cost':{},'budgets':{},'families':{}}
  for budget in [0,32,128,512]:
   result['budgets'][str(budget)]={}
   vals={form:[gap(r,form,budget) for r in rows] for form in ['shortest','edge_disjoint']};vals['routed']=[gap(r,f,budget) for r,f in zip(rows,rforms)]
   for form,x in vals.items():
    result['budgets'][str(budget)][form]={'mean_log1p_gap':loss(x),'median_gap':float(np.median(x)),'p90_gap':float(np.quantile(x,.9)),
     'median_gap_exact':float(np.median([gap(r,f if form=='routed' else form,budget,'gap_exact') for r,f in zip(rows,rforms)]))}
   if split=='holdout' and budget==128:
    rng=np.random.default_rng(2026100199);sample=rng.integers(0,len(rows),size=(2000,len(rows)))
    for base in ['shortest','edge_disjoint']:
     delta=np.log1p(vals['routed'])-np.log1p(vals[base]);ci=np.quantile(np.mean(delta[sample],axis=1),[.025,.975])
     result['budgets'][str(budget)]['routed']['difference_vs_'+base]={'mean':float(delta.mean()),'bootstrap_95_ci':ci.tolist(),'wins':sum(a<b for a,b in zip(vals['routed'],vals[base])),'losses':sum(a>b for a,b in zip(vals['routed'],vals[base]))}
  for form in ['shortest','edge_disjoint','routed']:
   runs=[r['forms'][f if form=='routed' else form] for r,f in zip(rows,rforms)]
   result['cost'][form]={k:float(sum(r[k] for r in runs)) for k in ['startup_seconds','wall_seconds','verification_seconds','cpu_seconds']}
   result['cost'][form]['first_sign']={str(q):{'median_censored_ordinal':float(np.median([r['first_sign'][str(q)] if r['first_sign'][str(q)] is not None else 513 for r in runs])),'decided':sum(r['first_sign'][str(q)] is not None for r in runs)} for q in RATIOS}
  for family in FAMILIES:
   subset=[r for r in rows if r['family']==family];forms=[router['below'] if feature(r,router['statistic'])<=router['threshold'] else router['above'] for r in subset]
   result['families'][family]={f:loss([gap(r,k if f=='routed' else f) for r,k in zip(subset,forms)]) for f in ['shortest','edge_disjoint','routed']}
  result['reference_total_seconds']=sum(r['reference_seconds'] for r in rows)
  result['statistics_median_fraction_of_reference']=float(np.median([r['forms']['shortest']['statistics_seconds']/r['reference_seconds'] for r in rows]))
  result['extra_cost_median_fraction_of_reference']={k:float(np.median([r['extra_statistics_cost'][k]/r['reference_seconds'] for r in rows])) for k in ['menger_seconds','spectral_seconds']}
  result['containment_checks']=sum(f['containment_checks'] for r in rows for f in r['forms'].values())
  result['run_receipt']={k:data[k] for k in ['wall_seconds','cpu_seconds','peak_rss_kib']}
  o['splits'][split]=result
 (DATA/'ANALYSIS.json').write_text(json.dumps(o,indent=2)+'\n')
 for split,r in o['splits'].items():print(split,json.dumps(r['budgets']['128']), 'checks',r['containment_checks'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);a=p.parse_args();train() if a.mode=='train' else evaluate()
