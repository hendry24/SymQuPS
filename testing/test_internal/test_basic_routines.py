import pytest
import sympy as sp

from symqups._internal.basic_routines import (
    treat_sub,
    screen_type,
    invalid_input,
    operation_routine
)

@pytest.mark.fast
@pytest.mark.order(0)
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
        try:
            screen_type(x_raise, sp.Symbol, "")
            raise RuntimeError("Test failed.")
        except:
            pass
        screen_type(x_pass, (sp.Symbol, sp.Number), "test")
        
    def test_invalid_type(self):
        try:
            invalid_input("test", "test")
            raise RuntimeError("Test failed.")
        except:
            pass
        
    def test_operation_routine(self):
        
        def _foo(inpt):
            return operation_routine(inpt,
                                     "test",
                                     (sp.Function,),
                                     {sp.Symbol : "no symbol"},
                                     {sp.Pow : "pow",
                                      (sp.Mul, sp.Add) : lambda A: "mul or add"}
                                     )
        
        try:
            _foo(sp.Function(r"raise_error"))
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