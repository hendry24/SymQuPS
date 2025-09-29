import pytest
import random
import sympy as sp

from symqups._internal.grouping import PhaseSpaceObject, qpType, alphaType

from symqups.objects.scalars import q, p,alpha, alphaD, W, t
from symqups.objects.operators import annihilateOp, createOp
from symqups.objects import scalars
from symqups.star import _Primed, _deprime, Star
from symqups.manipulations import qp2alpha
from symqups import s,hbar,zeta

hbar = hbar.val
zeta = zeta.val

@pytest.mark.fast
class TestStarProduct():
    
    rand_N = random.randint(1, 20)
    s.val = random.uniform(-1, 1)
    
    x = sp.Symbol("x")
    qq = q(rand_N)
    qq_prime = _Primed(qq)
    pp = p(rand_N)
    pp_prime = _Primed(pp)
    a = alpha(rand_N)
    a_prime = _Primed(a)
    ad = alphaD(rand_N)
    ad_prime = _Primed(ad)

    def test_Primed(self):
        for obj in [alpha(), alphaD(), annihilateOp(), createOp()]:
            assert isinstance(_Primed(obj), _Primed)
            assert _deprime(_Primed(obj)) == obj
        assert (_Primed(2*alpha()*sp.Symbol("x"))
                == 2*sp.Symbol("x")*_Primed(alpha()))
        
        assert not(isinstance(_Primed(W), _Primed))
        for arg in _Primed(W).args:
            if arg == t():
                continue
            if not(isinstance(arg, _Primed)):
                raise TypeError("_Primed(W) phase space variables must be _Primed.")

    def test_star(self):
        assert Star() == 1
        assert Star(*[1]*5)
        assert Star(self.x) == self.x
        assert isinstance(Star(scalars.W, scalars.W), Star)
        for n in range(1, 5):
            res = Star(*[self.a]*n)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res - self.a**n) == 0
            