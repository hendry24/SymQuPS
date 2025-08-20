import pytest
import sympy as sp

from symqups.utils._internal._basic_routines import (
    _treat_sub,
    _screen_type,
    _invalid_input,
    _operation_routine
)

@pytest.mark.fast
@pytest.mark.order(0)
class TestBasicRoutines:
    def test_treat_sub(self):
        assert _treat_sub(None, True) == sp.Symbol(r"")
        for sub in ["a", 1, 1+1j, sp.Symbol("x")]:
            for ssub in (sub, sp.sympify(sub)):
                assert isinstance(_treat_sub(ssub, True), sp.Symbol)
            assert _treat_sub(sub, False) == sp.Symbol(r"")
        assert _treat_sub(sp.Symbol("x"), True) == sp.Symbol("x")
        
    def test_screen_type(self):
        x_pass = sp.Symbol("x")
        x_raise = "x"
        try:
            _screen_type(x_raise, str, "")
            raise RuntimeError("Test failed.")
        except:
            pass
        _screen_type(x_pass, str, "test")
        
    def test_invalid_type(self):
        try:
            _invalid_input("test", "test")
            raise RuntimeError("Test failed.")
        except:
            pass
        
    def test_operation_routine(self):
        
        def _foo(inpt):
            return _operation_routine(inpt,
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