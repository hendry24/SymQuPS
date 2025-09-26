import pytest
import sympy as sp
import random

from symqups.objects.scalars import q, p, alpha, alphaD
from symqups.objects.operators import qOp, pOp, annihilateOp, createOp

from symqups.ordering import sOrdering
from symqups.manipulations import qp2alpha, express

from symqups import s as CahillGlauberS

# TESTED FUNCTIONALITIES
# ======================

from symqups.cg import CGTransform, iCGTransform

###

N = random.randint(1, 5)
n = random.random()
m = random.random()

CahillGlauberS.val = n

def template(inputs, outputs, foo):
    assert foo(sp.Number(1)) == sp.Number(1)
   
    q_in, p_in, a_in, ad_in = inputs
    q_out, p_out, a_out, ad_out = outputs
    
    assert not(foo(q_in*p_in).has(q, p))
    
    assert foo(a_in) == a_out
    assert foo(a_in**N) == a_out**N
    
    assert foo(ad_in) == ad_out
    assert foo(ad_in**N) == ad_out**N
    
    assert sp.simplify(sp.expand(foo(q_in) - qp2alpha(q_out))) == 0
    assert sp.simplify(sp.expand(foo(p_in) - qp2alpha(p_out))) == 0
    
@pytest.mark.fast
def test_CGTransform():
    template([qOp(), pOp(), annihilateOp(), createOp()],
             [q(), p(), alpha(), alphaD()],
             CGTransform)
    
    A = CGTransform(sOrdering(annihilateOp()*createOp()))
    B = alpha()*alphaD()
    assert sp.simplify(A-B) == 0

    A = CGTransform(sOrdering(annihilateOp()*createOp(), s=m))
    B = alpha()*alphaD()
    assert sp.simplify(A-B) != 0

@pytest.mark.fast
def test_iCGTransform():
    template([q(), p(), alpha(), alphaD()],
             [qOp(), pOp(), annihilateOp(), createOp()],
             iCGTransform)
    
    A = express(iCGTransform(alpha()*alphaD()), 1, explicit=True)
    B = express(sOrdering(annihilateOp()*createOp()), 1, explicit=True)
    assert sp.simplify(A-B) == 0