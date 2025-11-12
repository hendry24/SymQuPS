import pytest
import sympy as sp

from utils import expected_to_fail

# TESTED FUNCTIONALITIES
# ======================

from symqups._internal.basic_routines import (
    treat_sub,
    screen_type,
    deep_screen_type,
    invalid_input,
    operation_routine,
    only_allow_leaves_in_branches
)

@pytest.mark.fast
class TestBasicRoutines:
    def test_treat_sub(self):
        assert treat_sub(None, True) == sp.Symbol(r"")
        for sub in ["a", 1, 1+1j, sp.Symbol("x")]:
            for ssub in (sub, sp.sympify(sub)):
                assert isinstance(treat_sub(ssub, True), sp.Symbol)
            assert treat_sub(sub, False) == sp.Symbol(r"")
        assert treat_sub(sp.Symbol("x"), True) == sp.Symbol("x")
        
    def test_screen_type(self):
        x_pass = sp.Add(sp.Symbol("x"), 2)
        x_raise = sp.Symbol("x")
        expected_to_fail(lambda : screen_type(x_raise, sp.Symbol, ""))
        screen_type(x_pass, (sp.Symbol, sp.Number), "test")
        
    def test_deep_screen_type(self):
        x_raise_1 = sp.Symbol("x")
        x_raise_2 = sp.Add(sp.Symbol("x"), 2)
        x_pass = sp.Number(1)
        for xx in [x_raise_1, x_raise_2]:
            expected_to_fail(lambda: deep_screen_type(xx, sp.Symbol, ""))
        screen_type(x_pass, (sp.Symbol, sp.Function), "test")
        
    def test_invalid_type(self):
        expected_to_fail(lambda: invalid_input("test", "test"))
        
    def test_operation_routine(self):
        
        def _foo(inpt):
            return operation_routine(inpt,
                                     "test",
                                     [sp.Function,],
                                     [sp.Function,],
                                     {sp.Symbol : "no symbol"},
                                     {sp.Pow : "pow",
                                      (sp.Mul, sp.Add) : lambda A: "mul or add"}
                                     )
        
        try:
            _foo(sp.Function(r"raise_error"))
            raise RuntimeError("Test failed.")
        except:
            pass
        
        try:
            _foo(sp.Add(sp.Function(r"raise_error_as_well"), sp.Number(2)))
            raise RuntimeError("Test failed.")
        except:
            pass
        
        inputs = [sp.Expr(sp.Number(1)), # no symbol,  
                  sp.Symbol("x")**2, 
                  sp.Symbol("x") * 2,
                  sp.Symbol("x") + 2]
        expected_outputs = ["no symbol", 
                            "pow", 
                            "mul or add",
                            "mul or add"]
        for inpt, e_out in zip(inputs[:1], expected_outputs[:1]):
            assert _foo(inpt) == e_out
            
    def test_only_allow_leaves_in_branches(self):
        from symqups.objects.scalars import q, p
        q = q()
        p = p()
        only_allow_leaves_in_branches(q,"",p.func,sp.Pow)
        only_allow_leaves_in_branches(q**2,"",q.func,sp.Pow)
        try:
            only_allow_leaves_in_branches(q,"",q.func,sp.Pow)
            raise RuntimeError("Test failed.")
        except:
            pass
        
        only_allow_leaves_in_branches(q*p+2*p,"",(q.func,p.func),sp.Mul)
        try:
            only_allow_leaves_in_branches(q**2+q*p**2, "", (q.func,p.func), sp.Mul)
            raise RuntimeError("Test failed.")
        except:
            pass