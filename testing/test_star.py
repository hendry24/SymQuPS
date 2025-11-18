import pytest
import random
import sympy as sp

from symqups._internal.grouping import PhaseSpaceObject, qpType, alphaType

from symqups.objects.scalars import q, p,alpha, alphaD, W, t
from symqups.objects.operators import annihilateOp, createOp, rho
from symqups.objects import scalars
from symqups.star import Star, HattedStar
from symqups.manipulations import Derivative, Commutator, normal_ordered_equivalent
from symqups import s,hbar,zeta
from symqups.utils import get_random_poly

hbar = hbar.val
zeta = zeta.val
s = s.val

@pytest.mark.fast
class TestStar():
    
    rand_N = random.randint(1, 20)
    
    x = sp.Symbol("x")
    qq = q(rand_N)
    pp = p(rand_N)
    a = alpha(rand_N)
    ad = alphaD(rand_N)
    aOp = annihilateOp(rand_N)
    adOp = createOp(rand_N)

    def test_Star(self):
        assert Star() == 1
        assert Star(0, 1, self.ad) == 0
        assert Star(*[1]*5) == 1
        assert Star(self.x) == self.x
        assert Star(self.aOp) == self.aOp
        assert sp.expand(Star(self.a+self.ad, self.a).doit()- (self.a**2 + self.ad*self.a+(s-1)/2)) == 0
        for n in range(1, 3):
            res = Star(*[self.a]*n)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res.doit() - self.a**n) == 0
        assert Star(self.a, W) == (self.a*W + (s+1)/2*Derivative(W, self.ad)).doit().expand()
        assert Star(self.ad, W) == (self.ad*W + (s-1)/2*Derivative(W, self.a)).doit().expand()
        assert Star(W, self.a) == (self.a*W + (s-1)/2*Derivative(W, self.ad)).doit().expand()
        assert Star(W, self.ad) == (self.ad*W + (s+1)/2*Derivative(W, self.a)).doit().expand()
        assert (Star(self.a, self.ad)- Star(self.ad, self.a)).doit().expand() == 1
        
        foo = Star(sp.exp(self.ad), W)
        assert isinstance(foo, Star) and foo.args == (sp.exp(self.ad), W)
        
        ins = [get_random_poly([self.a,self.ad]) for _ in range(3)]
        
        # Associativity
        assert (Star(ins[0], Star(ins[1], ins[2])) 
                - Star(Star(ins[0], ins[1]), ins[2])).expand() == 0
        
        # Distributivity over first argument
        assert (Star(ins[0]+ins[1], ins[2])
                - Star(ins[0], ins[2])
                - Star(ins[1], ins[2])).expand() == 0
        
        # Distributivity over second argument
        assert (Star(ins[0], ins[1]+ins[2])
                - Star(ins[0], ins[1])
                - Star(ins[0], ins[2])).expand() == 0
        
    def test_HattedStar(self):
        assert HattedStar(1) == 1
        assert HattedStar(*[1]*5) == 1
        assert HattedStar(self.a) == self.a
        assert HattedStar(self.x) == self.x
        for n in range(1,3):
            res = HattedStar(*[self.aOp]*n)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res - self.aOp**n) == 0
        X = HattedStar(self.adOp, rho)
        Xe = self.adOp*rho - sp.Rational(1,2)*(s-1)*Commutator(rho, self.adOp)
        Y = HattedStar(rho, self.adOp)
        Ye = rho*self.adOp - sp.Rational(1,2)*(s+1)*Commutator(rho, self.adOp)
        assert (X-Xe).doit().expand() == 0
        assert (Y-Ye).doit().expand() == 0
        assert (X-Y).doit().expand() == 0
        assert isinstance(HattedStar(sp.exp(self.adOp), rho), HattedStar)
        
        ins = [get_random_poly([self.aOp, self.adOp]) for _ in range(3)]
        
        # Associativity
        assert normal_ordered_equivalent(
                HattedStar(ins[0], HattedStar(ins[1], ins[2])) 
                - HattedStar(HattedStar(ins[0], ins[1]), ins[2])).expand() == 0
        
        # Distributivity over first argument
        assert normal_ordered_equivalent(
            (HattedStar(ins[0]+ins[1], ins[2])
                - HattedStar(ins[0], ins[2])
                - HattedStar(ins[1], ins[2])).expand()).expand() == 0
        
        # Distributivity over second argument
        assert normal_ordered_equivalent(
            (HattedStar(ins[0], ins[1]+ins[2])
                - HattedStar(ins[0], ins[1])
                - HattedStar(ins[0], ins[2])).expand()).expand() == 0