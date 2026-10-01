import subprocess,sys,json,resource
from pathlib import Path
from time import perf_counter,process_time
from experiment import LANE,DATA
start=perf_counter();cpu=process_time();runs=[]
commands=[['successors.py','stars'],['experiment.py','leaves'],['router_replay.py'],['path_exposure.py'],['-m','pytest','-q','worktree/tests/test_certified_resistance.py','worktree/tests/test_instance_slack.py']]
for i,cmd in enumerate(commands):
 t=perf_counter();name=['stars','leaves','router_replay','path_exposure','delivery_tests'][i]
 with (LANE/(name+'.log')).open('w') as log:
  r=subprocess.run([sys.executable,*cmd],cwd=LANE,stdout=log,stderr=subprocess.STDOUT,env={**__import__('os').environ,'PYTHONPATH':str(LANE/'worktree/src')})
 runs.append({'command':[sys.executable,*cmd],'returncode':r.returncode,'wall_seconds':perf_counter()-t,'log':name+'.log'})
 print(name,r.returncode,perf_counter()-t,flush=True)
 assert r.returncode==0,cmd
receipt={'runs':runs,'wall_seconds':perf_counter()-start,'cpu_seconds_parent':process_time()-cpu,'peak_rss_children_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
(DATA/'REMAINING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
