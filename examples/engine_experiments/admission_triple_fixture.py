"""Constructed admission fixture. These labels are NOT empirical graph truth."""
import json
from pathlib import Path

def registry(target_ids=None):
 targets=list(target_ids) if target_ids is not None else ['positive-control']
 return {'source-A':{'verified':True,'source_family':['family-A'],'supports':targets},
         'source-B':{'verified':True,'source_family':['family-B'],'supports':targets}}

def calibration():
 n=128
 return {'scores':[[10.0 if i%8==2 else 9.0 if i%8 in [0,1] else i/(2*n) for i in range(n)],
                   [10.0 if i%8==3 else 9.0 if i%8 in [0,1] else i/(2*n) for i in range(n)]],
         'selected':[i%8 in [0,1] for i in range(n)],'true_labels':[int(i%8 in [0,1]) for i in range(n)],
         'candidate_ids':[f'c-{i:04d}' for i in range(n)],'source_families':[f'f-{i//8:03d}' for i in range(n)],
         'n_bootstrap':200,'seed':20261001}

def template(root, identity='positive-control', metric=0.375):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
 art=root/'positive-control.json'
 obj={'measured':metric,'agreement_calibration':calibration(),
      'validation':{'instrument_family':'predictor','reference_family':'independent-observer',
                    'event_basis':'external_observation','test_relation':'independent_oracle'},
      'source_reports':[{'source_id':'source-A'}, {'source_id':'source-B'}],'inherited_support':0}
 art.write_text(json.dumps(obj))
 script=root/'replay_fixture.py'
 # This checks the fixture and never claims experimental remeasurement.
 script.write_text("import json\nfrom pathlib import Path\nx=json.loads((Path(__file__).parent/'positive-control.json').read_text())\nassert abs(x['measured']-0.375)<1e-10\n")
 return {'cell':identity,'node':identity,'verdict':'positive','rerun':{'command':f'python {script}', 'artifact':str(art),
        'checks':[{'key':'measured','expected':metric,'tol':1e-10}]},
        'ATOMS':{'report':'fixture','claims':[{'id':'measured','text':'fixture metric preserved','load_bearing':True}],
                 'atoms':[{'id':'metric','claim':'measured','type':'value-in-artifact','artifact':str(art),
                           'key':'measured','expected':metric,'tol':1e-10}]}}
