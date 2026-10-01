import subprocess,sys,json,resource,os
from pathlib import Path
from time import perf_counter,process_time
from experiment import LANE,DATA
start=perf_counter();cpu=process_time();runs=[]
for name,args in [('final_analysis',['final_analysis.py']),('delivery_tests',['-m','pytest','-q','worktree/tests/test_certified_resistance.py','worktree/tests/test_instance_slack.py']),('exact_audit',['final_audit.py'])]:
 t=perf_counter()
 with (LANE/(name+'.log')).open('w') as log:
  r=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,cwd=LANE,env={**os.environ,'PYTHONPATH':str(LANE/'worktree/src')})
 runs.append({'command':[sys.executable,*args],'returncode':r.returncode,'wall_seconds':perf_counter()-t,'log':name+'.log'})
 print(name,r.returncode,flush=True);assert r.returncode==0
(DATA/'FINAL_CHECKS.json').write_text(json.dumps({'runs':runs,'wall_seconds':perf_counter()-start,'cpu_seconds_parent':process_time()-cpu,'peak_rss_children_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss},indent=2)+'\n')
