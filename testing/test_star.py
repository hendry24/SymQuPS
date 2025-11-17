import pytest
import random
import sympy as sp

from symqups._internal.grouping import PhaseSpaceObject, qpType, alphaType

from symqups.objects.scalars import q, p,alpha, alphaD, W, t
from symqups.objects.operators import annihilateOp, createOp, rho
from symqups.objects import scalars
from symqups.star import Star, HattedStar
from symqups.manipulations import Derivative, Commutator
from symqups import s,hbar,zeta

hbar = hbar.val
zeta = zeta.val

@pytest.mark.fast
class TestStar():
    
    rand_N = random.randint(1, 20)
    s.val = random.uniform(-1, 1)
    s_val = s.val
    
    x = sp.Symbol("x")
    qq = q(rand_N)
    pp = p(rand_N)
    a = alpha(rand_N)
    ad = alphaD(rand_N)
    aOp = annihilateOp(rand_N)
    adOp = createOp(rand_N)

    def test_Star(self):
        s = self.s_val
        assert Star() == 1
        assert Star(*[1]*5) == 1
        assert Star(self.x) == self.x
        assert Star(self.aOp) == self.aOp
        for n in range(1, 3):
            res = Star(*[self.a]*n)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res.doit() - self.a**n) == 0
        assert Star(self.a, W) == self.a*W + (s+1)/2*Derivative(W, self.ad)
        assert Star(W, self.a) == self.a*W + (s-1)/2*Derivative(W, self.ad)
        assert (Star(self.a, self.ad)- Star(self.ad, self.a) - 1).doit().expand() == 0
        assert isinstance(Star(sp.exp(self.ad), W), Star)
        
    def test_HattedStar(self):
        s = self.s_val
        assert HattedStar(1) == 1
        assert HattedStar(*[1]*5) == 1
        assert HattedStar(self.a) == self.a
        assert HattedStar(self.x) == self.x
        for n in range(1,3):
            res = HattedStar(*[self.aOp]*n)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res - self.aOp**n) == 0
        X = HattedStar(self.adOp, rho)
        Y = HattedStar(rho, self.adOp)
        assert X == self.adOp*rho - sp.Rational(1,2)*(s-1)*Commutator(rho, self.adOp)
        assert Y == rho*self.adOp - sp.Rational(1,2)*(s+1)*Commutator(rho, self.adOp)
        assert (X-Y).doit().expand() == 0
        assert isinstance(HattedStar(sp.exp(self.adOp), rho), HattedStar)