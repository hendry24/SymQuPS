import pytest
import dill
import random
import sympy as sp

from symqups.objects.scalars import (hbar, pi, mu, Scalar, q, p, t, W, alpha, alphaD,
                                    _Primed, _DePrimed, _DerivativeSymbol, StateFunction)
from symqups.objects.operators import (Operator, qOp, pOp, createOp, annihilateOp,
                                        densityOp, rho, Dagger)

from symqups.objects.cache import _sub_cache

def get_random_poly(objects, coeffs=[1], max_pow=3, dice_throw=10):
    """
    Make a random polynomial in 'objects'.
    """
    return sp.Add(*[sp.Mul(*[random.choice(coeffs)*random.choice(objects)**random.randint(0, max_pow)
                             for _ in range(dice_throw)])
                    for _ in range(dice_throw)])

def arithmetic(A):
    A+2
    2+A
    A-2
    2-A
    A*2
    2*A
    A/2
    2/A
    A**(-2)
    assert A**(0) == 1
    A**2
    sp.sqrt(A)
    sp.sin(A)
    sp.exp(A)
    sp.log(A)

@pytest.mark.order(0)
class TestScalars():
    
    def test_scalar_construction(self):
        for sub in [None, "1", 1, sp.Number(1), sp.Symbol("1")]:
            obj = Scalar(sub)
            assert isinstance(obj.sub, sp.Symbol)
            assert obj.sub in _sub_cache
            assert dill.loads(dill.dumps(obj)) == obj
        
        arithmetic(obj)
        
        for base, obj in zip(["t", "q", "p"], 
                             [t(), q(), p()]):
            assert isinstance(obj, Scalar)
            assert base in sp.latex(obj)
            assert obj.is_Atom
    
    def test_alpha(self):
        a_sc = alpha()
        assert a_sc.is_Atom
        assert r"\alpha" in sp.latex(a_sc)
        a_sc_expanded = (q()*mu + sp.I*p()/mu) / sp.sqrt(2*hbar)
        assert (a_sc.define() - a_sc_expanded).expand() == 0

        ad_sc = alphaD()
        assert ad_sc.is_Atom
        assert r"\alpha^*" in sp.latex(ad_sc)
        ad_sc_expanded = (q()*sp.conjugate(mu) - sp.I*p()/sp.conjugate(mu)) / sp.sqrt(2*hbar)
        assert (ad_sc.define() - ad_sc_expanded).expand() == 0
        
        assert sp.conjugate(alpha()) == alphaD()
        assert (sp.conjugate(alpha().define()) - alphaD().define()).expand() == 0
        assert sp.conjugate(alphaD()) == alpha()
        assert (sp.conjugate(alphaD().define()) - alpha().define()).expand() == 0
    
    def test_primed(self):
        assert _Primed(q()).atoms(_Primed)
        assert not(_Primed(q()).is_commutative)
        
        rand_poly = get_random_poly(objects=[q(), p(), alpha(), alphaD(), sp.Symbol("x")],
                                    coeffs=[1, sp.Symbol(r"\kappa"), sp.exp(-sp.I/2*sp.Symbol(r"\Gamma"))])
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
        assert isinstance(W(), StateFunction)
        assert isinstance(W(), sp.Function)
        check_vars = [t()]
        global _sub_cache
        for sub in _sub_cache:
            check_vars.extend([q(sub), p(sub)])
        assert W().free_symbols == set(check_vars)
        W_str = sp.latex(W(False))
        assert ("W_s" in W_str and
                "q" not in W_str and
                "p" not in W_str)
        W_str = sp.latex(W(True))
        assert ("W_s" in W_str and
                "q" in W_str and
                "p" in W_str)
        assert W(False) == W(True)
        
@pytest.mark.order(1)
class TestHilbertOps():
    def test_operator_construction(self):
        for sub in [None, "1", 1, sp.Number(1), sp.Symbol("1")]:
            obj = Operator(sub)
            assert isinstance(obj.sub, sp.Symbol)
            assert dill.loads(dill.dumps(obj)) == obj
        
        arithmetic(obj)
        
        for base, obj in zip([r"\hat{q}", r"\hat{p}", 
                              r"\hat{a}", r"\hat{a}^{\dagger}",
                              r"\rho"], 
                             [qOp(), pOp(), 
                              annihilateOp(), createOp(),
                              densityOp()]):
            assert isinstance(obj, Operator)
            assert obj.is_Atom
            assert base in sp.latex(obj)
            
        assert rho() == densityOp()
    
    def test_dagger(self):
        for herm_op in [qOp(), pOp(), densityOp()]:
            assert Dagger(herm_op) == herm_op

        assert Dagger(annihilateOp()) == createOp()
        assert (Dagger(annihilateOp().define())-createOp().define()).expand() == 0
        assert Dagger(createOp()) == annihilateOp()            
        assert (Dagger(createOp().define())-annihilateOp().define()).expand() == 0

        rand_poly = get_random_poly(objects = (1, sp.Symbol("x"), qOp(), annihilateOp(),
                                               createOp(), annihilateOp()),
                                    coeffs = list(range(10)) + sp.symbols([]))
        assert (Dagger(Dagger(rand_poly)) - rand_poly).expand() == 0