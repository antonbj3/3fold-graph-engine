import json
from fractions import Fraction as F
from experiment import DATA,graph,FAMILIES,oracle
from f1_experiment import exact_sparse
receipts=[]
for family in FAMILIES:
 for seed in [20261001,20261002]:
  e,a,b=graph(family,seed)
  ref,r=exact_sparse(e,a,b)
  assert ref==oracle.independent_oracle(e,a,b)
  receipts.append({'family':family,'seed':seed,'same_exact_fraction':True})
old=json.loads((DATA/'F1_DECIMAL_DIAGNOSTIC.json').read_text())['rows'][0]['reference']
from pathlib import Path
line=Path('exact_pilot.log').read_text().split()[1]
assert F(old)==F(line)
print('20 flint sparse references match independent Fraction Gaussian elimination; full-bank Fraction pilot retained.')
(DATA/'FLINT_CHECK.json').write_text(json.dumps({'checks':receipts,'full_bank_fraction_pilot_matches_archived_reference':True},indent=2)+'\n')
