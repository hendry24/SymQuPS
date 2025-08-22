import pytest
import sympy as sp

from symqups.objects.operators import Operator, createOp, annihilateOp
from symqups.utils._internal._operator_handling import (
    _separate_operator,
    _separate_term_by_polynomiality,
    _separate_term_oper_by_sub,
    _collect_alpha_type_oper_from_monomial_by_sub
)
from symqups.objects import cache

@pytest.mark.fast
@pytest.mark.order(5)
class TestOperatorHandling:
    def test_separate_operator(self):
        op_1 = Operator(1)
        op_2 = Operator(2)
        x = sp.Symbol("x")
        foo = sp.Function("f")(x, op_1)
        
        try:
            _separate_term_by_polynomiality(x+2)
            raise RuntimeError("Test failed.")
        except:
            pass

        assert _separate_operator(op_1) == (1, op_1)
        assert _separate_operator(x) == (x, 1)
        assert _separate_operator(x * op_1) == (x, op_1)
        
        assert _separate_operator(x**2) == (x**2, 1)
        assert _separate_operator(op_2**2) == (1, op_2**2)
        
        assert (_separate_operator(op_1*foo*op_2**2*x**3*2) 
                == (2*x**3, op_1*foo*op_2**2))
        
    def test_separate_term_by_polynomiality(self):
        a_1 = annihilateOp(1)
        ad_2 = createOp(2)
        x = sp.Symbol("x")
        
        try:
            _separate_term_by_polynomiality(a_1+1)
            raise RuntimeError("Test failed.")
        except:
            pass
        
        assert (_separate_term_by_polynomiality(a_1)
                == [a_1])
        assert (_separate_term_by_polynomiality(a_1**2*ad_2)
                == [a_1**2*ad_2])
        assert (_separate_term_by_polynomiality(a_1**2*ad_2*sp.exp(x))
                == [a_1**2*ad_2*sp.exp(x)])
        assert (_separate_term_by_polynomiality(a_1*x*sp.exp(ad_2)*ad_2**3 * a_1**0.3 * 2**ad_2)
                == [a_1*x, sp.exp(ad_2), ad_2**3, a_1**0.3*2**ad_2])
        
    def test_collect_alpha_type_oper_from_monomial_by_sub(self):
        cache._sub_cache.clear()
        
        a_1 = annihilateOp(1)
        ad_1 = createOp(1)
        a_2 = annihilateOp(2)
        ad_2 = createOp(2)
        x = sp.Symbol("x")
        
        try:
            _separate_term_by_polynomiality(a_1+1)
            raise RuntimeError("Test failed.")
        except:
            pass
        
        try:
            _separate_term_by_polynomiality(sp.exp(ad_2))
            raise RuntimeError("Test failed.")
        except:
            pass
        
        assert _collect_alpha_type_oper_from_monomial_by_sub(x)[0] == x
        
        col_ad = _collect_alpha_type_oper_from_monomial_by_sub(ad_1)[1]
        assert isinstance(col_ad, dict)
        assert col_ad[ad_1.sub] == [ad_1, 1]
        
        col_a = _collect_alpha_type_oper_from_monomial_by_sub(a_1)[2]
        assert isinstance(col_a, dict)
        assert col_a[a_1.sub] == [a_1, 1]
        
        expr = 2*x*sp.exp(x) * a_2**2 * a_1 * a_2 * a_1 * ad_2**3 * a_2**1
        non_op, col_ad, col_a = _collect_alpha_type_oper_from_monomial_by_sub(expr)
        
        assert non_op == 2*x*sp.exp(x)
        assert col_a.pop(a_1.sub) == [a_1, 2]
        assert col_ad.pop(ad_1.sub) == [ad_1, 0]
        assert col_a.pop(a_2.sub) == [a_2, 4]
        assert col_ad.pop(ad_2.sub) == [ad_2, 3]
        assert not(col_ad)
        assert not(col_a)
    
    def test_separate_term_oper_by_sub(self):
        
        foo = _separate_term_oper_by_sub
        
        assert foo(1) == [1]
        
        a = [annihilateOp(i) for i in range(4)]
        ad = [annihilateOp(i) for i in range(4)]
        x = sp.Symbol("x")
        
        try:
            _collect_alpha_type_oper_from_monomial_by_sub(a[0]+ad[0])
            raise RuntimeError("Test failed.")
        except:
            pass
        
        assert foo(a[0]) == [a[0]]
        assert foo(a[0]*a[1]) == list(sp.ordered([a[0], a[1]]))
        assert foo(a[0]*sp.exp(a[0]*a[1])*a[1]) == [a[0]*sp.exp(a[0]*a[1])*a[1]]
        assert (foo(2*x * a[0] * ad[1]*2**a[1] * sp.log(a[2]+a[3]**2)*a[3]**3 * a[1])
                == [2*x, a[0], ad[1]*2**a[1]*a[1], sp.log(a[2]+a[3]**2)*a[3]**3])
                # May raise an error if sympy's ordering is not its current canon. 