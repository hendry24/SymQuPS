import pytest
import dill
import random
import sympy as sp

from moyalstar.core.scalars import (hbar,pi, Scalar, q, p, t, W, alpha, alphaD,
                                    _Primed, _DePrimed, _DerivativeSymbol, WignerFunction)
from moyalstar.core.hilbert_operators import (Operator, qOp, pOp, createOp, annihilateOp,
                                        densityOp, rho, Dagger)
from moyalstar.core.star_product import (Bopp, Star, _star_base,
                                         _first_index_and_diff_order, _replace_diff)

from moyalstar.core.base import _sub_cache
from moyalstar.utils.multiprocessing import _mp_helper, MP_CONFIG

def get_random_poly(objects, coeffs=[1], max_pow=3, dice_throw=10):
    """
    Make a random polynomial in 'objects'.
    """
    return sp.Add(*[sp.Mul(*[random.choice(coeffs)*random.choice(objects)**random.randint(0, max_pow)
                             for _ in range(dice_throw)])
                    for _ in range(dice_throw)])

@pytest.mark.order(0)
class TestScalars():
    global _sub_cache
    
    def test_scalar_construction(self):
        for sub in [None, "1", 1, sp.Number(1), sp.Symbol("1")]:
            obj = Scalar(sub)
            assert isinstance(obj.sub, sp.Symbol)
            assert obj.sub in _sub_cache
            assert dill.loads(dill.dumps(obj)) == obj
        
        for base, obj in zip(["t", "q", "p"], [t(), q(), p()]):
            assert isinstance(obj, Scalar)
            assert base in sp.latex(obj)
    
    def test_alpha(self):
        a_sc = alpha()
        a_sc_expanded = (q() + sp.I*p()) / sp.sqrt(2*hbar)
        assert (a_sc - a_sc_expanded).expand() == 0

        ad_sc = alphaD()
        ad_sc_expanded = (q() - sp.I*p()) / sp.sqrt(2*hbar)
        assert (ad_sc - ad_sc_expanded).expand() == 0
    
    def test_primed(self):
        rand_poly = get_random_poly(objects=[q(), p(), alpha(), alphaD(), sp.Symbol("x")],
                                    coeffs=[1, sp.Symbol(r"\kappa"), sp.exp(-sp.I/2*sp.Symbol(r"\Gamma"))])
        assert _Primed(rand_poly).atoms(_Primed)
        assert not(_Primed(sp.I*2*sp.Symbol("x")).atoms(_Primed))
        assert (_DePrimed(_Primed(rand_poly)) - rand_poly).expand() == 0
        assert not(_Primed(rand_poly).is_commutative)
        assert (_DePrimed(_Primed(rand_poly)) - rand_poly).expand() == 0

    def test_derivative_symbol(self):
        try:
            _DerivativeSymbol(q())
            raise TypeError("Input must be _Primed.")
        except:
            pass
        der = _DerivativeSymbol(_Primed(q()))
        assert isinstance(der.diff_var, _Primed)
        assert not(der.is_commutative)
        
    def test_W(self):
        assert isinstance(W(), WignerFunction)
        assert isinstance(W(), sp.Function)
        check_vars = [t()]
        global _sub_cache
        for sub in _sub_cache:
            check_vars.extend([q(sub), p(sub)])
        assert W().free_symbols == set(check_vars)
        W_str = sp.latex(W)
        assert ("W" in W_str and
                "q" not in W_str and
                "p" not in W_str)
        
@pytest.mark.order(1)
class TestHilbertOps():
    def test_operator_construction(self):
        for sub in [None, "1", 1, sp.Number(1), sp.Symbol("1")]:
            obj = Operator(sub)
            assert isinstance(obj.sub, sp.Symbol)
            assert dill.loads(dill.dumps(obj)) == obj
        
        for base, obj in zip([r"\hat{q}", r"\hat{p}", 
                              r"\hat{a}", r"\hat{a}^{\dagger}",
                              r"\rho"], 
                             [qOp(), pOp(), 
                              annihilateOp(), createOp(),
                              densityOp()]):
            assert isinstance(obj, Operator)
            assert base in sp.latex(obj)
            
        assert rho() == densityOp()
    
    def test_dagger(self):
        assert Dagger(annihilateOp()) == createOp()
        assert Dagger(createOp()) == annihilateOp()

        for herm_op in [qOp(), pOp(), densityOp()]:
            assert Dagger(herm_op) == herm_op
            
        rand_poly = get_random_poly(objects = (1, sp.Symbol("x"), qOp(), annihilateOp(),
                                               createOp(), annihilateOp()),
                                    coeffs = list(range(10)) + sp.symbols([]))
        assert (Dagger(Dagger(rand_poly)) - rand_poly).expand() == 0
    
    def test_wigner_transform(self):
        N = len(_sub_cache)
        for op, wig in zip([qOp(), pOp(),
                            createOp(), annihilateOp(),
                            densityOp()],
                           [q(), p(), 
                            alphaD(), alpha(), 
                            (2*pi*hbar)**N*W()]):
            assert (op.wigner_transform() - wig).expand() == 0

def mp_helper_foo(x):
        return x+2
@pytest.mark.order(2)
def test_mp_helper():
    inpt = [1, sp.Symbol("x"), Scalar(),
            Operator(), W()]
    
    global MP_CONFIG
    enable_default = MP_CONFIG["enable"]
    MP_CONFIG["min_num_args"] = 0

    for enable in [True, False]:
        MP_CONFIG["enable"] = enable
        assert (_mp_helper(inpt, mp_helper_foo) 
                == list(map(mp_helper_foo, inpt)))
    
    MP_CONFIG["enable"] = enable_default

@pytest.mark.order(3)
class TestStarProduct():
    
    rand_N = random.randint(0, 100)
    
    x = sp.Symbol("x")
    q = q(rand_N)
    qq = _Primed(q)
    dqq = _DerivativeSymbol(qq)
    p = p(rand_N)
    pp = _Primed(p)
    dpp = _DerivativeSymbol(pp)
    a = alpha(rand_N)
    ad = alphaD(rand_N)
        
    def test_bopp_shift(self):
        q_bopp_right =  Bopp(q(), left=False)
        q_bopp_left = Bopp(q(), left=True)
        p_bopp_right = Bopp(p(), left=False)
        p_bopp_left = Bopp(p(), left=True)
        ddq = _DerivativeSymbol(_Primed(q()))
        ddp = _DerivativeSymbol(_Primed(p()))
        
        for bopped, check in zip([q_bopp_right, 
                                  q_bopp_left, 
                                  p_bopp_right, 
                                  p_bopp_left],
                                 [q() + sp.I*hbar/2*ddp,
                                  q() - sp.I*hbar/2*ddp,
                                  p() - sp.I*hbar/2*ddq,
                                  p() + sp.I*hbar/2*ddq]):
            assert (bopped - check).expand() == 0
            assert not(bopped.is_commutative)
            
    def test_fido(self):
        
        def FIDO(x):
            return _first_index_and_diff_order(x)
        
        try:
            FIDO(self.x+2+self.dqq)
            raise ValueError("Input should be invalid.")
        except:
            pass
        
        assert FIDO(1*self.x*self.q*self.pp) is None
        assert FIDO(self.qq**5) is None
       
        assert FIDO(self.dqq) == (0, self.qq, 1)
        assert FIDO(self.dpp) == (0, self.pp, 1)
        
        assert FIDO(self.dqq**self.rand_N) == (0, self.qq, self.rand_N)
        assert FIDO(self.dpp**self.rand_N) == (0, self.pp, self.rand_N)
    
        assert FIDO(self.qq**5*self.p*self.dqq*self.pp) == (2, self.qq, 1)
        assert FIDO(self.dpp*self.dqq) == (0, self.pp, 1)
        
        random_symbols = [sp.Symbol(r"TEST-{%s}" % n, commutative=False) for n in range(100)]
        random_symbols[self.rand_N] = self.dqq
        assert FIDO(sp.Mul(*random_symbols)) == (self.rand_N, self.qq, 1)
        
    def test_replace_diff(self):
        WW = _Primed(W())
        
        assert _replace_diff(sp.Integer(1)) == 1
        assert _replace_diff(self.x) == self.x
        assert _replace_diff(self.dqq) == sp.Derivative(1, self.qq, evaluate=False)
        
        assert _replace_diff(self.dqq*WW) == sp.Derivative(WW, self.qq)
        assert _replace_diff(self.dpp*WW) == sp.Derivative(WW, self.pp)
        
        assert (_replace_diff(self.dqq**2*self.dpp*WW) 
                == sp.Derivative(sp.Derivative(WW, self.pp), 
                                 self.qq, 2, evaluate=False))
        
        assert (_replace_diff(self.dqq*self.qq*self.pp*WW) 
                == sp.Derivative(self.qq*self.pp*WW, self.qq, evaluate=False))
        
    def test_star_base(self):
        def must_raise_error(bad_A, bad_B):
            try:
                _star_base(bad_A, bad_B)
                raise ValueError("Input should be invalid.")
            except:
                pass    
        for bad_A, bad_B in [[sp.sqrt(self.q), sp.sqrt(self.p)],
                             [sp.Function("foo_A")(self.q, self.p), W()],
                             [self.q**0.2, self.p**1.0000]]:
            must_raise_error(bad_A, bad_B)
        
        q0, p0, a0, ad0 = self.q, self.p, self.a, self.ad
        q1, p1, a1, ad1 = q(self.rand_N+1), p(self.rand_N+1), alpha(self.rand_N+1), alphaD(self.rand_N+1)
        for A, B, out in [[q0, q0, q0**2],
                          [p0, p0, p0**2],
                          [q0, p0, p0*q0 + sp.I*hbar/2],
                          [p0, q0, p0*q0 - sp.I*hbar/2],
                          [a0, ad0, (q0**2+p0**2+hbar)/(2*hbar)],
                          [ad0, a0, (q0**2+p0**2-hbar)/(2*hbar)],
                          [q0, p1, q0*p1],
                          [p0, q1, p0*q1],
                          [a0, ad1, a0*ad1],
                          [ad0, a1, ad0*a1]]:
            
            assert (_star_base(A, B) - out).expand() == 0
        
    def test_star(self):
        assert Star() == 1
        assert Star(self.q) == self.q
        for n in range(2, 5):
            assert Star(*[self.q]*n) == self.q**n