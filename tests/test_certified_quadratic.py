from dataclasses import replace
from fractions import Fraction as F
import random
import pytest
from graph_engine.certified_quadratic import certified_quadratic_decision,verify_quadratic_witness
from graph_engine._rational_interface import solve_right


@pytest.mark.parametrize('seed',range(30))
def test_positive_offdiagonal_exact_quadratic(seed):
    rng=random.Random(202610017500+seed);n=8
    J=[[F(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):J[i][j]=J[j][i]=F(rng.randint(-3,5),7)
    for i in range(n):J[i][i]=F(1,3)+sum(abs(c) for c in J[i])
    h=[F(rng.randint(-2,2)) for _ in range(n)]
    x=solve_right(J,[[c] for c in h]);q=sum(c*y[0] for c,y in zip(h,x))
    for ratio in [F(1,2),F(99,100),F(101,100),F(2)]:
        result=certified_quadratic_decision(J,h,q*ratio,30)
        w=result['witness'];assert w.lower<=q<=w.upper
        verify_quadratic_witness(J,h,q*ratio,result)
        if q:assert result['verdict']==('GREATER' if ratio<1 else 'LESS')
        changed=[row.copy() for row in J];changed[0][0]+=1
        with pytest.raises(ValueError,match='hash'):verify_quadratic_witness(changed,h,q*ratio,result)


def test_missing_spectral_bound_abstains_by_rejection():
    with pytest.raises(ValueError,match='Gershgorin'):
        certified_quadratic_decision([[1,1],[1,1]],[1,-1],0,10)


def test_forged_quadratic_bound():
    J=[[F(2),F(1)],[F(1),F(2)]];h=[F(1),F(0)]
    result=certified_quadratic_decision(J,h,F(1),10)
    result['witness']=replace(result['witness'],upper=F(0))
    with pytest.raises(ValueError,match='bounds'):verify_quadratic_witness(J,h,F(1),result)
