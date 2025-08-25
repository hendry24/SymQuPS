import pytest
import random
import sympy as sp

from symqups._internal.grouping import PhaseSpaceObject, qpType, alphaType
from symqups.objects.scalars import q, p,_Primed, _DerivativeSymbol, alpha, alphaD, hbar, mu
from symqups.objects import scalars
from symqups.star_product import Bopp, _first_index_and_diff_order, _replace_diff, _star_base, Star
from symqups.manipulations import qp2a
from symqups import s

@pytest.mark.fast
class TestStarProduct():
    
    rand_N = random.randint(1, 20)
    s.val = random.uniform(-1, 1)
    
    x = sp.Symbol("x")
    qq = q(rand_N)
    qq_prime = _Primed(qq)
    dqq = _DerivativeSymbol(qq_prime)
    pp = p(rand_N)
    pp_prime = _Primed(pp)
    dpp = _DerivativeSymbol(pp_prime)
    a = alpha(rand_N)
    a_prime = _Primed(a)
    da = _DerivativeSymbol(a_prime)
    ad = alphaD(rand_N)
    ad_prime = _Primed(ad)
    dad = _DerivativeSymbol(ad_prime)
        
    def test_bopp_shift(self):
        qq_br =  Bopp(self.qq, left=False)
        qq_bl = Bopp(self.qq, left=True)
        pp_br = Bopp(self.pp, left=False)
        pp_bl = Bopp(self.pp, left=True)
        a_br = Bopp(self.a, left=False)
        a_bl = Bopp(self.a, left=True)
        ad_br = Bopp(self.ad, left=False)
        ad_bl = Bopp(self.ad, left=True)
        
        for bopped, check in zip([qq_br, 
                                  qq_bl, 
                                  pp_br, 
                                  pp_bl,
                                  a_br, 
                                  a_bl,
                                  ad_br, 
                                  ad_bl
                                  ],
                                 [self.qq + sp.I*hbar/2*self.dpp + s.val*hbar/2*(1/mu**2)*self.dqq,
                                  self.qq - sp.I*hbar/2*self.dpp + s.val*hbar/2*(1/mu**2)*self.dqq,
                                  self.pp - sp.I*hbar/2*self.dqq + s.val*hbar/2*mu**2*self.dpp,
                                  self.pp + sp.I*hbar/2*self.dqq + s.val*hbar/2*mu**2*self.dpp,
                                  self.a + (s.val+1)/2*self.dad,
                                  self.a + (s.val-1)/2*self.dad,
                                  self.ad + (s.val-1)/2*self.da,
                                  self.ad + (s.val+1)/2*self.da
                                  ]):
            assert sp.expand(bopped - check) == 0
            assert not(bopped.is_commutative)
        
        assert not(Bopp(1).has(_DerivativeSymbol, PhaseSpaceObject))
        assert not(Bopp(self.x).has(_DerivativeSymbol, PhaseSpaceObject))
        
        assert isinstance(Bopp(sp.Derivative(scalars.W, self.a)), Bopp)
                    
    def test_fido(self):
        
        def FIDO(x):
            return _first_index_and_diff_order(x)
        
        try:
            FIDO(self.x+2+self.dqq)
            raise ValueError("Input should be invalid.")
        except:
            pass
        
        for primed in [self.qq_prime, self.pp_prime, self.a_prime, self.ad_prime]:
            assert FIDO(1*self.x*self.qq*self.pp_prime) is None
            assert FIDO(primed**5) is None
            assert FIDO(_DerivativeSymbol(primed)) == (0, primed, 1)
            assert FIDO(_DerivativeSymbol(primed)**self.rand_N) == (0, primed, self.rand_N)
            assert FIDO(_DerivativeSymbol(primed)*self.a_prime*self.dqq) == (0, primed, 1)
            assert FIDO(self.x*primed**5*_DerivativeSymbol(primed)*self.pp_prime) == (2, primed, 1)
            
        random_symbols = [sp.Symbol(r"TEST-{%s}" % n, commutative=False) for n in range(100)]
        random_symbols[self.rand_N] = self.dqq
        assert FIDO(sp.Mul(*random_symbols)) == (self.rand_N, self.qq_prime, 1)
        
    def test_replace_diff(self):
        WW = _Primed(scalars.W)
        
        for primed in [self.qq_prime, self.pp_prime, self.a_prime, self.ad_prime]:
            assert _replace_diff(sp.Integer(1)) == 1
            assert _replace_diff(self.x) == self.x
            assert _replace_diff(_DerivativeSymbol(primed)) == sp.Derivative(1, primed, evaluate=False)
            
            assert _replace_diff(_DerivativeSymbol(primed)*WW) == sp.Derivative(WW, primed)
                        
            assert (_replace_diff(_DerivativeSymbol(primed)**2*self.dpp*WW) 
                    == sp.Derivative(sp.Derivative(WW, self.pp_prime), 
                                    primed, 2, evaluate=False))
                    
    def test_star_base(self):
        def must_raise_error(bad_A, bad_B):
            try:
                _star_base(bad_A, bad_B)
                raise ValueError("Input should be invalid.")
            except:
                pass    
        for bad_A, bad_B in [[sp.sqrt(self.qq), sp.sqrt(self.pp)],
                             [sp.Function("foo_A")(self.qq, self.pp), scalars.W],
                             [self.qq**0.2, self.pp**1.0000]]:
            must_raise_error(bad_A, bad_B)
        
        q0, p0, a0, ad0 = self.qq, self.pp, self.a, self.ad
        q1, p1, a1, ad1 = q(self.rand_N+1), p(self.rand_N+1), alpha(self.rand_N+1), alphaD(self.rand_N+1)
        for A, B, out in [[q0, q0, qp2a(q0**2 + s.val*hbar/2*(1/mu**2))],
                          [p0, p0, qp2a(p0**2 + s.val*hbar/2*mu**2)],
                          [q0, p0, qp2a(p0*q0 + sp.I*hbar/2)],
                          [p0, q0, qp2a(p0*q0 - sp.I*hbar/2)],
                          [a0, ad0, a0*ad0+(s.val+1)/2],
                          [ad0, a0, ad0*a0+(s.val-1)/2],
                          [q0, p1, qp2a(q0*p1)],
                          [p0, q1, qp2a(p0*q1)],
                          [a0, ad1, a0*ad1],
                          [ad0, a1, ad0*a1]
                         ]:
            
            res = _star_base(A, B)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res - out) == 0
        
    def test_star(self):
        assert Star() == 1
        assert Star(*[1]*5)
        assert Star(self.x) == self.x
        assert isinstance(Star(scalars.W, scalars.W), Star)
        for n in range(1, 5):
            res = Star(*[self.a]*n)
            assert res.has(alphaType) and not(res.has(qpType))
            assert sp.expand(res - self.a**n) == 0