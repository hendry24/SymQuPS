import pytest
import random
import sympy as sp

from symqups._internal.grouping import PhaseSpaceObject, qpType, alphaType

from symqups.objects.scalars import q, p,alpha, alphaD, W, t
from symqups.objects.operators import annihilateOp, createOp
from symqups.objects import scalars
from symqups.star import Star
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
    pp = p(rand_N)
    a = alpha(rand_N)
    ad = alphaD(rand_N)

    def test_star(self):
        assert Star() == 1
        assert Star(*[1]*5)
        assert Star(self.x) == self.x
        assert isinstance(Star(scalars.W, scalars.W), Star)
        for n in range(1, 5):
            res = Star(*[self.a]*n)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res - self.a**n) == 0
            