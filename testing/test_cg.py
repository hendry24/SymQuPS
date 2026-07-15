import pytest
import sympy as sp
import random

from symqups.objects.scalars import q, p, alpha, alphaD, W, t
from symqups.objects.operators import qOp, pOp, annihilateOp, createOp, rho, TimeDependentOp

from symqups.ordering import sOrdering
from symqups.manipulations import qp2alpha, op2sc, Derivative, Commutator, normal_ordered_equivalent
from symqups.utils import get_random_poly, get_N
from symqups.star import Star, HattedStar

from symqups import s, pi
s = s.val
pi = pi.val

from symqups._internal.cache import sub_cache
sub_cache.clear()

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
    assert CGTransform(rho) == W
    
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
    
    # HattedStar product
    F = sp.Function("F")(a,ad)
    G = sp.Function("G")(a,ad)
    assert CGTransform(HattedStar(F,G)) == CGTransform(F)*CGTransform(G)
    
    # Derivative
    assert CGTransform(Derivative(rho, t())) == Derivative(W, t())
    
    # Commutator
    A = CGTransform(Commutator(ad*a, rho)).doit().expand()
    B = alphaD()*Derivative(W,alphaD()) - alpha()*Derivative(W,alpha())
    assert sp.expand(A-B) == 0
    
    # Check mode consistency
    A = CGTransform(ins[0], mode="Star")
    B = CGTransform(ins[0], mode="PSBO")
    C = CGTransform(ins[0], mode="explicit")
    
    assert (sp.expand(A-B) == 0
            and sp.expand(A-C) == 0
            and sp.expand(B-C) == 0)
    
    ###
    
    from symqups import s
    old_s_val = s.val
    
    # Test P correspondence
    s.val = -1
    aop = annihilateOp()
    adop = createOp()
    a = alpha()
    ad = alphaD()
    assert "P" in sp.latex(CGTransform(rho))
    assert CGTransform(aop*rho) == a*W
    assert sp.expand(CGTransform(adop*rho) - (ad*W -sp.Derivative(W,a))) == 0
    assert sp.expand(CGTransform(rho*aop) - (a*W - sp.Derivative(W,ad))) == 0
    assert CGTransform(rho*adop) == ad*W
    
    # Test W correspondence
    s.val = 0
    assert "W" in sp.latex(CGTransform(rho))
    assert sp.expand(CGTransform(aop*rho) - (a*W + sp.Derivative(W, ad)/2)) == 0
    assert sp.expand(CGTransform(adop*rho) - (ad*W - sp.Derivative(W, a)/2)) == 0
    assert sp.expand(CGTransform(rho*aop) - (a*W - sp.Derivative(W,ad)/2)) == 0
    assert sp.expand(CGTransform(rho*adop) - (ad*W + sp.Derivative(W,a)/2)) == 0
    
    # Test Q correspondence
    s.val = 1
    assert "Q" in sp.latex(CGTransform(rho))
    assert sp.expand(CGTransform(aop*rho) - (a*W + sp.Derivative(W, ad))) == 0
    assert sp.expand(CGTransform(adop*rho) - (ad*W)) == 0
    assert sp.expand(CGTransform(rho*aop) - (a*W)) == 0
    assert sp.expand(CGTransform(rho*adop) - (ad*W + sp.Derivative(W,a))) == 0
    
    # Return s value for other tests
    s.val = old_s_val

@pytest.mark.fast
def test_iCGTransform():
    template([q(), p(), alpha(), alphaD()],
             [qOp(), pOp(), annihilateOp(), createOp()],
             iCGTransform)
    
    a, ad = alpha(), alphaD()
    
    ins = [get_random_poly([a,ad]) for _ in range(2)]
    
    # State function
    assert iCGTransform(W) == rho
    
    # Addition
    A = iCGTransform(ins[0]+ins[1])
    B = iCGTransform(ins[0]) + iCGTransform(ins[1])
    assert sp.expand(normal_ordered_equivalent(A-B)) == 0
    
    # Derivative
    assert iCGTransform(Derivative(W, a)) == Commutator(rho, createOp())
    assert iCGTransform(Derivative(W, ad)) == Commutator(annihilateOp(), rho)
    assert iCGTransform(Derivative(W, t())).doit() == Derivative(rho, t())
    
    # Power
    assert isinstance(iCGTransform(Derivative(W,t())**2), iCGTransform)
    assert isinstance(iCGTransform(W**2), iCGTransform)
    assert not isinstance(iCGTransform(a**2), iCGTransform)
    
    # Mul
    A = normal_ordered_equivalent(iCGTransform(ins[0]*ins[1]))
    B = normal_ordered_equivalent(HattedStar(iCGTransform(ins[0]), iCGTransform(ins[1])))
    assert sp.expand(A-B) == 0
    
    # Function
    foo = sp.Function("f")
    assert isinstance(iCGTransform(foo(W)), iCGTransform)
    assert not isinstance(iCGTransform(foo(a,ad)), iCGTransform)
    assert iCGTransform(foo(a,ad)) == sOrdering(foo(annihilateOp(), createOp()))
    
    # Star product
    F = sp.Function("F")(a,ad)
    G = sp.Function("G")(a,ad)
    assert iCGTransform(Star(F,G)) == iCGTransform(F)*iCGTransform(G)
    
    ###
    
    from symqups import s
    old_s_val = s.val
    
    # Test P correspondence
    s.val = -1
    aop = annihilateOp()
    adop = createOp()
    a = alpha()
    ad = alphaD()
    assert "P" in sp.latex(W)
    assert CGTransform(aop*rho) == a*W
    assert sp.expand(adop*rho - iCGTransform(ad*W -sp.Derivative(W,a)).doit()) == 0
    assert sp.expand(rho*aop - iCGTransform(a*W - sp.Derivative(W,ad)).doit()) == 0
    assert rho*adop == iCGTransform(ad*W)
    
    # Test W correspondence
    s.val = 0
    assert "W" in sp.latex(W)
    assert sp.expand((aop*rho) -  iCGTransform(a*W + sp.Derivative(W, ad)/2).doit()) == 0
    assert sp.expand((adop*rho) - iCGTransform (ad*W - sp.Derivative(W, a)/2).doit()) == 0
    assert sp.expand((rho*aop) -  iCGTransform(a*W - sp.Derivative(W,ad)/2).doit()) == 0
    assert sp.expand((rho*adop) - iCGTransform (ad*W + sp.Derivative(W,a)/2).doit()) == 0
    
    # Test Q correspondence
    s.val = 1
    assert "Q" in sp.latex(W)
    assert sp.expand((aop*rho) - iCGTransform(a*W + sp.Derivative(W, ad)).doit()) == 0
    assert sp.expand((adop*rho) -iCGTransform (ad*W).doit()) == 0
    assert sp.expand((rho*aop) - iCGTransform(a*W).doit()) == 0
    assert sp.expand((rho*adop) -iCGTransform (ad*W + sp.Derivative(W,a)).doit()) == 0
    
    # Return s value for other tests
    s.val = old_s_val
    
pytest.mark.fast
def test_CG_correspondence():
    a, ad = annihilateOp(), createOp()
    rp = get_random_poly([a,ad])
    
    assert normal_ordered_equivalent(iCGTransform(CGTransform(rp)) - rp).expand() == 0
    
    A = Commutator(ad*a, rho).doit().expand()
    B = normal_ordered_equivalent(iCGTransform(CGTransform(A))).doit()
    assert sp.expand(A-B) == 0