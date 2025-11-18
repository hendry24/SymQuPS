import pytest
import dill
import sympy as sp

# TESTED FUNCTIONALITIES
# ======================

from symqups._internal.cache import sub_cache
from symqups._internal.grouping import PhaseSpaceVariable, qpType, alphaType

from symqups.objects.scalars import (Scalar, t, q, p, alpha, alphaD, StateFunction, W)
from symqups.objects.operators import Operator, qOp, pOp, createOp, annihilateOp, densityOp, rho, TimeDependentOp

from symqups._internal.cache import sub_cache
sub_cache.clear()

###

@pytest.mark.fast
class TestObjectInstantiation:
    def test_Scalar(self):
        for sub in [None, "1", 1, sp.Number(1), sp.Symbol("1")]:
            obj = Scalar(sub)
            assert isinstance(obj.sub, sp.Symbol)
            assert obj.sub in sub_cache
            assert dill.loads(dill.dumps(obj)) == obj
        
        for base, obj in zip(["t", "q", "p"], 
                             [t(), q(), p()]):
            assert isinstance(obj, Scalar)
            assert base in sp.latex(obj)
            assert obj.is_Atom
        
        for obj in [q(), p()]:
            assert isinstance(obj, PhaseSpaceVariable)
            assert isinstance(obj, qpType)
            
        for obj in [alpha(), alphaD()]:
            assert obj.is_Atom
            assert isinstance(obj, PhaseSpaceVariable)
            assert isinstance(obj, alphaType)
            assert r"\alpha" in sp.latex(obj) 

    def test_W(self):
        
        sub_cache.clear()
        check_vars = [t()]
        for sub in [1,2,3]:
            check_vars.extend([alpha(sub), alphaD(sub)])
            q(sub), p(sub)
        
        assert set(sp.ordered(W.args)) == set(sp.ordered(check_vars))
        
        assert isinstance(W, StateFunction)
        assert isinstance(W, sp.Expr)
        assert not(isinstance(W, PhaseSpaceVariable))
        assert ("alpha" in sp.latex(W.args)
                and "q_" not in sp.latex(W.args)
                and "p_" not in sp.latex(W.args))
                
        q(r"newly_added_sub")
        assert "newly_added_sub" in sp.latex(W.args)
        
    def test_Operator(self):
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
            assert obj.is_Atom
            assert base in sp.latex(obj)
            
        assert rho == densityOp()
        assert not(rho.has_sub)
        
        A = TimeDependentOp(rho)
        assert isinstance(A, TimeDependentOp)
        assert dill.loads(dill.dumps(A)) == A