import pytest
import random
import sympy as sp

from symqups.objects.scalars import alpha, alphaD, W
from symqups.objects.operators import annihilateOp, createOp, rho
from symqups.manipulations import Commutator

from symqups import s
s = s.val

from symqups._internal.cache import sub_cache
sub_cache.clear()

from utils import expected_to_fail

# TESTED FUNCTIONALITIES
# ======================

from symqups.bopp import PSBO, HSBS, PhaseSpaceBoppOperator, HilbertSpaceBoppSuperoperator

###

rand_sub = random.randint(0, 100)
a = alpha(rand_sub)
ad = alphaD(rand_sub)
aop = annihilateOp(rand_sub)
adop = createOp(rand_sub)

@pytest.mark.fast
def test_PSBO():
    
    assert PSBO == PhaseSpaceBoppOperator
    
    def foo():
        PSBO(1)
        PSBO(rho)
        PSBO(W)
        PSBO(aop)
        PSBO(adop)
    expected_to_fail(foo)
    
    assert PSBO(a).base == a
    assert PSBO(ad).base == ad
    
    assert not(isinstance(PSBO(a, 1), 
                          PSBO))
    
    assert PSBO(a, a, True).doit() == a**2
    assert PSBO(a, a, False).doit() == a**2
    assert sp.expand(PSBO(a, ad, True).doit() 
                     - (a*ad + (s-1)/2)) == 0
    assert sp.expand(PSBO(a, ad, False).doit()
                     - (a*ad + (s+1)/2)) == 0
    assert sp.expand(PSBO(a, W, True).doit() 
                     - (a*W + (s-1)/2 * sp.Derivative(W, ad))) == 0
    assert sp.expand(PSBO(a, W, False).doit()
                     - (a*W + (s+1)/2 * sp.Derivative(W, ad))) == 0
    
    assert sp.expand(PSBO(ad, a, True).doit() 
                     - (ad*a + (s+1)/2)) == 0
    assert sp.expand(PSBO(ad, a, False).doit()
                     - (ad*a + (s-1)/2)) == 0
    assert PSBO(ad, ad, True).doit() == ad**2
    assert PSBO(ad, ad, False).doit() == ad**2
    assert sp.expand(PSBO(ad, W, True).doit() 
                     - (ad*W + (s+1)/2*sp.Derivative(W, a))) == 0
    assert sp.expand(PSBO(ad, W, False).doit()
                     - (ad*W + (s-1)/2*sp.Derivative(W, a))) == 0
    
    assert sp.expand(PSBO(a).act(ad).doit() - PSBO(a,ad).doit()) == 0

###
    
@pytest.mark.fast
def test_HSBS():
    
    assert HSBS == HilbertSpaceBoppSuperoperator
    
    def foo():
        HSBS(1)
        HSBS(rho)
        HSBS(W)
        HSBS(a)
        HSBS(ad)
    expected_to_fail(foo)
    
    assert HSBS(aop).base == aop
    assert HSBS(adop).base == adop
    
    assert not(isinstance(HSBS(aop, 1), 
                          HSBS))
    
    assert HSBS(aop, aop, True).doit() == aop**2
    assert HSBS(aop, aop, False).doit() == aop**2
    assert sp.expand(HSBS(aop, adop, True).doit() 
                     - (adop*aop - (s-1)/2*Commutator(aop,adop).doit())) == 0
    assert sp.expand(HSBS(aop, adop, False).doit()
                     - (aop*adop - (s+1)/2*Commutator(aop,adop).doit())) == 0
    assert sp.expand(HSBS(aop, rho, True).doit()
                     - (rho*aop - (s-1)/2*Commutator(aop, rho).doit())) == 0
    assert sp.expand(HSBS(aop, rho, False).doit()
                     - (aop*rho - (s+1)/2*Commutator(aop, rho).doit())) == 0
    
    assert sp.expand(HSBS(adop, aop, True).doit() 
                     - (aop*adop - (s+1)/2*Commutator(aop, adop).doit())) == 0
    assert sp.expand(HSBS(adop, aop, False).doit()
                     - (adop*aop - (s-1)/2*Commutator(aop,adop).doit())) == 0
    assert HSBS(adop, adop, True).doit() == adop**2
    assert HSBS(adop, adop, False).doit() == adop**2
    assert sp.expand(HSBS(adop, rho, True).doit()
                     - (rho*adop - (s+1)/2*Commutator(rho, adop).doit())) == 0
    assert sp.expand(HSBS(adop, rho, False).doit()
                     - (adop*rho - (s-1)/2*Commutator(rho, adop).doit())) == 0
    
    assert sp.expand(HSBS(aop).act(adop).doit() - HSBS(aop,adop).doit()) == 0
