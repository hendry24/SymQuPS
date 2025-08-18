import pytest
import dill
import sympy as sp

from symqups.objects.base import PhaseSpaceObject, qpTypePSO, alphaTypePSO
from symqups.objects.scalars import (hbar, mu, Scalar, q, p, t, W, alpha, alphaD,
                                    _Primed, _DePrimed, _DerivativeSymbol, StateFunction)
from symqups.objects.operators import (Operator, qOp, pOp, createOp, annihilateOp,
                                        densityOp, rho, Dagger)

from symqups.objects.cache import _sub_cache
from symqups.utils.algebra import get_random_poly
from symqups.utils._internal import _treat_sub

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

@pytest.mark.fast
@pytest.mark.order(0)
class TestScalars():
    
    def test_treat_sub_and_scalar_construction(self):
        for sub in [None, "1", 1, sp.Number(1), sp.Symbol("1")]:
            assert _treat_sub(_treat_sub(sub, True), True) == _treat_sub(sub, True)
            assert _treat_sub(_treat_sub(sub, False), False) == _treat_sub(sub, False)
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
        
        for obj in [q(), p()]:
            assert isinstance(obj, PhaseSpaceObject)
            assert isinstance(obj, qpTypePSO)
            
        assert not(_treat_sub("xxx", True) == t("xxx").sub)
    
    def test_alpha(self):

        for obj, expanded in zip([alpha(), alphaD()],
                                 [(q()*mu + sp.I*p()/mu) / sp.sqrt(2*hbar),
                                  (q()*sp.conjugate(mu) - sp.I*p()/sp.conjugate(mu)) / sp.sqrt(2*hbar)]):
            assert obj.is_Atom
            assert isinstance(obj, PhaseSpaceObject)
            assert isinstance(obj, alphaTypePSO)
            assert r"\alpha" in sp.latex(obj) 
            assert sp.expand(obj.define() - expanded) == 0
        
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

@pytest.mark.fast
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