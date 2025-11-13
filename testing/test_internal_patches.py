import pytest
import sympy as sp

from symqups.objects.scalars import alpha, alphaD, t, W
from symqups.objects.operators import annihilateOp, createOp, rho

@pytest.mark.fast
class TestInternalPatches:
    a = [alpha(i) for i in range(4)]
    ad = [alphaD(i) for i in range(4)]
    aOp = [annihilateOp(i) for i in range(4)]
    adOp = [createOp(i) for i in range(4)]
    #
    def test_Mul(self):
        a, ad, aOp, adOp = self.a, self.ad, self.aOp, self.adOp
        
        assert a[0]*ad[1] == ad[1]*a[0]
        assert ((a[0]*ad[1]**2*sp.exp(a[0])).args
                == (ad[1]**2, a[0], sp.exp(a[0])))
        
        assert aOp[0]*aOp[1] == aOp[1]*aOp[0]
        assert ((aOp[0]*sp.exp(aOp[2])*sp.exp(aOp[0]*aOp[1])*aOp[3]*adOp[1]).args
                == (aOp[0], sp.exp(aOp[0]*aOp[1]), adOp[1], sp.exp(aOp[2]), aOp[3]))
        
        assert ((aOp[0]*aOp[1]*rho*aOp[1]*adOp[0]).args
                == (aOp[0], aOp[1], rho, adOp[0], aOp[1]))
        
    def test_Derivative(self):
        a, ad, aOp, adOp = self.a, self.ad, self.aOp, self.adOp
        tt = t()
        x = sp.Symbol("x")
        F = sp.Function("F")(*a, *ad, *aOp, *adOp, x)
 
        for der in [sp.Derivative(F, a[0]),
                    sp.Derivative(rho, tt),
                    sp.Derivative(W, tt)]:
            assert der.doit() == der