import pytest
import sympy as sp
import random

from symqups.objects.scalars import q, p, alpha, alphaD, W, t
from symqups.objects.operators import qOp, pOp, annihilateOp, createOp, rho, _TimeDependentOp

from symqups.ordering import sOrdering
from symqups.manipulations import qp2alpha, express_sOrdering, op2sc, Derivative, Commutator
from symqups.utils import get_random_poly
from symqups.star import Star, HattedStar

from symqups import s, pi
s = s.val
pi = pi.val

# TESTED FUNCTIONALITIES
# ======================

from symqups.cg import CGTransform, iCGTransform

###

N = random.randint(1, 5)
n = random.random()
m = random.random()

def template(inputs, outputs, foo):
    assert foo(sp.Number(1)) == sp.Number(1)
    assert foo(1+1j) == sp.sympify(1+1j)
   
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
    
    a, ad = annihilateOp(), createOp()
    
    ins = [get_random_poly([a, ad]) for _ in range(3)]
    
    # No operator
    A = sp.Symbol("x")*alpha()
    assert CGTransform(A) == A
    
    # Density matrix
    assert CGTransform(rho) == pi*W
    
    # Addition
    A = CGTransform(ins[0]+ins[1])
    B = CGTransform(ins[0]) + CGTransform(ins[1])
    assert sp.expand(A-B) == 0
    
    # Multiplication
    A = CGTransform(ins[0]*ins[1])
    B = Star(CGTransform(ins[0]), CGTransform(ins[1]))
    assert sp.expand(A-B) == 0
    
    # sOrdering
    assert CGTransform(sOrdering(a*ad)) == op2sc(a*ad)
    
    # Functions
    foo = sp.Function("f")
    assert not isinstance(CGTransform(foo(a)), CGTransform)
    assert not isinstance(CGTransform(foo(ad)), CGTransform)
    assert isinstance(CGTransform(foo(a, ad)), CGTransform)
    assert isinstance(CGTransform(foo(rho)), CGTransform)
    
    # HattedStar
    F = sp.Function("F")(a,ad)
    G = sp.Function("G")(a,ad)
    assert CGTransform(HattedStar(F,G)) == CGTransform(F)*CGTransform(G)
    
    # Derivative
    assert CGTransform(Derivative(_TimeDependentOp(rho)/pi, t())) == Derivative(W, t())
    
    # Commutator
    A = CGTransform(Commutator(ad*a, rho/pi)).doit().expand()
    B = alphaD()*Derivative(W,alphaD()) - alpha()*Derivative(W,alpha())
    assert sp.expand(A-B) == 0
    
    # Check mode consistency
    A = CGTransform(ins[0], mode="Star")
    B = CGTransform(ins[0], mode="PSBO")
    C = CGTransform(ins[0], mode="explicit")
    
    assert (sp.expand(A-B) == 0
            and sp.expand(A-C) == 0
            and sp.expand(B-C) == 0)

@pytest.mark.fast
def test_iCGTransform():
    template([q(), p(), alpha(), alphaD()],
             [qOp(), pOp(), annihilateOp(), createOp()],
             iCGTransform)
    
    A = express_sOrdering(iCGTransform(alpha()*alphaD()), 1, explicit=True)
    B = express_sOrdering(sOrdering(annihilateOp()*createOp()), 1, explicit=True)
    assert sp.simplify(A-B) == 0
    
    