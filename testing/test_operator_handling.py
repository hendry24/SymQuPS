import pytest
import sympy as sp

from symqups.objects.operators import Operator, createOp, annihilateOp, rho
from symqups._internal.cache import sub_cache

# TESTED FUNCTIONALITIES
# ======================

from symqups._internal.operator_handling import (
    get_oper_sub,
    is_universal,
    separate_operator,
    separate_term_by_polynomiality,
    separate_term_oper_by_sub,
    collect_alpha_type_oper_from_monomial_by_sub
)

@pytest.mark.fast
class TestOperatorHandling:
    def test_get_oper_sub(self):
        assert len(get_oper_sub(rho)) == 0
        assert get_oper_sub(createOp(1)) == {createOp(1).sub}
        sub_cache.clear()
        a = [annihilateOp(i) for i in range(4)]
        ad = [createOp(i) for i in range(4)]
        assert get_oper_sub(sp.Mul(*a, *ad)) == set(sub_cache)
        
    def is_universal(self):
        try:
            is_universal(1)
            raise RuntimeError("Test failed")
        except:
            pass
        
        assert is_universal(rho) == True
        assert is_universal(createOp()) == False
        assert is_universal(rho*createOp() == True)
    
    def test_separate_operator(self):
        op_1 = Operator(1)
        op_2 = Operator(2)
        x = sp.Symbol("x")
        foo = sp.Function("f")(x, op_1)
        
        try:
            separate_term_by_polynomiality(x+2)
            raise RuntimeError("Test failed.")
        except:
            pass

        assert separate_operator(op_1) == (1, op_1)
        assert separate_operator(x) == (x, 1)
        assert separate_operator(x * op_1) == (x, op_1)
        
        assert separate_operator(x**2) == (x**2, 1)
        assert separate_operator(op_2**2) == (1, op_2**2)
        
        assert (separate_operator(op_1*foo*op_2**2*x**3*2) 
                == (2*x**3, op_1*foo*op_2**2))
        
    def test_separate_term_by_polynomiality(self):
        a_1 = annihilateOp(1)
        ad_2 = createOp(2)
        x = sp.Symbol("x")
        
        try:
            separate_term_by_polynomiality(a_1+1)
            raise RuntimeError("Test failed.")
        except:
            pass
        
        assert (separate_term_by_polynomiality(a_1)
                == [a_1])
        assert (separate_term_by_polynomiality(a_1**2*ad_2)
                == [a_1**2*ad_2])
        assert (separate_term_by_polynomiality(a_1**2*ad_2*sp.exp(x))
                == [a_1**2*ad_2*sp.exp(x)])
        assert (separate_term_by_polynomiality(a_1*x*sp.exp(ad_2)*ad_2**3 * a_1**0.3 * 2**ad_2)
                == [x, a_1**1.3, sp.exp(ad_2), ad_2**3, 2**ad_2])
        
    def test_collect_alpha_type_oper_from_monomial_by_sub(self):
        sub_cache.clear()
        
        a_1 = annihilateOp(1)
        ad_1 = createOp(1)
        a_2 = annihilateOp(2)
        ad_2 = createOp(2)
        x = sp.Symbol("x")
        
        try:
            separate_term_by_polynomiality(a_1+1)
            raise RuntimeError("Test failed.")
        except:
            pass
        
        try:
            separate_term_by_polynomiality(sp.exp(ad_2))
            raise RuntimeError("Test failed.")
        except:
            pass
        
        assert collect_alpha_type_oper_from_monomial_by_sub(x)[0] == x
        
        col_ad = collect_alpha_type_oper_from_monomial_by_sub(ad_1)[1]
        assert isinstance(col_ad, dict)
        assert col_ad[ad_1.sub] == [ad_1, 1]
        
        col_a = collect_alpha_type_oper_from_monomial_by_sub(a_1)[2]
        assert isinstance(col_a, dict)
        assert col_a[a_1.sub] == [a_1, 1]
        
        expr = 2*x*sp.exp(x) * a_2**2 * a_1 * a_2 * a_1 * ad_2**3 * a_2**1
        non_op, col_ad, col_a = collect_alpha_type_oper_from_monomial_by_sub(expr)
        
        assert non_op == 2*x*sp.exp(x)
        assert col_a.pop(a_1.sub) == [a_1, 2]
        assert col_ad.pop(ad_1.sub) == [ad_1, 0]
        assert col_a.pop(a_2.sub) == [a_2, 4]
        assert col_ad.pop(ad_2.sub) == [ad_2, 3]
        assert not(col_ad)
        assert not(col_a)
    
    def test_separate_term_oper_by_sub(self):
        
        foo = separate_term_oper_by_sub
        
        assert foo(sp.Number(1)) == [1]
        
        a = [annihilateOp(i) for i in range(4)]
        ad = [annihilateOp(i) for i in range(4)]
        x = sp.Symbol("x")
        
        try:
            collect_alpha_type_oper_from_monomial_by_sub(a[0]+ad[0])
            raise RuntimeError("Test failed.")
        except:
            pass
        
        assert foo(a[0]) == [1, a[0]]
        assert (list(sp.ordered(foo(a[0]*a[1]))) 
                == list(sp.ordered([1, a[0], a[1]])))
        assert (list(sp.ordered(foo(a[0]*sp.exp(a[0]*a[1])*a[1]))) 
                == list(sp.ordered([1, a[0]*sp.exp(a[0]*a[1])*a[1]])))
        
        assert (list(sp.ordered(foo(2*x * a[0] * ad[1]*2**a[1] * sp.log(a[2]+a[3]**2)*a[3]**3 * a[1])))
                == list(sp.ordered([2*x, a[0], ad[1]*2**a[1]*a[1], sp.log(a[2]+a[3]**2)*a[3]**3])))
                # May raise an error if sympy's ordering is not its current canon. 