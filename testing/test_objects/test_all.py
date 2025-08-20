import pytest
import dill
import sympy as sp

from symqups.objects.base import PhaseSpaceObject, qpTypePSO, alphaTypePSO
from symqups.objects.scalars import (hbar, mu, Scalar, q, p, t, W, alpha, alphaD,
                                    _Primed, _DePrimed, _DerivativeSymbol, StateFunction)
from symqups.objects.operators import (Operator, qOp, pOp, createOp, annihilateOp,
                                        densityOp, rho)

from symqups.objects.cache import _sub_cache
from symqups.utils.algebra import get_random_poly
from symqups.utils._internal._basic_routines import _treat_sub

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
@pytest.mark.order(1)
class TestScalars():
    
    def and_scalar_construction(self):
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
        from symqups.objects.scalars import W
        
        check_vars = [t()]
        global _sub_cache
        for sub in _sub_cache:
            check_vars.extend([alpha(sub), alphaD(sub)])
        assert W.free_symbols == set(check_vars)
        
        assert isinstance(W, StateFunction)
        assert isinstance(W, sp.Function)
        assert ("alpha" not in sp.latex(W)
                and "q_" not in sp.latex(W)
                and "p_" not in sp.latex(W))
        W.show_vars = True
        assert ("alpha" in sp.latex(W)
                and "q_" not in sp.latex(W)
                and "p_" not in sp.latex(W))
        
        p(r"newly_added_sub")
        
        from symqups.objects.scalars import W
        W.show_vars = True
        assert "newly_added_sub" in sp.latex(W)
        
@pytest.mark.fast
@pytest.mark.order(2)
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
            
        assert rho == densityOp()