import pytest
import sympy as sp

from symqups.objects.operators import Operator, createOp, annihilateOp, rho
from symqups.objects.scalars import alpha, alphaD
from symqups._internal.cache import sub_cache
from symqups._internal.grouping import PhaseSpaceVariableOperator

from utils import expected_to_fail

from symqups._internal.cache import sub_cache
sub_cache.clear()

# TESTED FUNCTIONALITIES
# ======================

from symqups._internal.math import (
    get_sub,
    is_nonconstant_polynomial,
    is_pip,
    separate_term_by_nonconstant_polynomiality,
    get_factor_nonconstant_polynomiality,
    has_universal_oper,
    separate_operator,
    collect_alpha_type_oper_from_monomial_by_sub,
    separate_term_oper_by_sub
)

###

@pytest.mark.fast
class TestInternalMath:
    def test_get_sub(self):
        for a, ad in [[annihilateOp, createOp],
                      [alpha, alphaD]]:
            assert len(get_sub(rho)) == 0
            assert get_sub(ad(1)) == {ad(1).sub}
            sub_cache.clear()
            aa = [a(i) for i in range(4)]
            aad = [ad(i) for i in range(4)]
            assert get_sub(sp.Mul(*aa, *aad)) == set(sub_cache)
    
    def test_is_nonconstant_polynomial(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        assert not is_nonconstant_polynomial(sp.Number(1), x)
        assert not is_nonconstant_polynomial(x, y)
        assert is_nonconstant_polynomial(x, x)
        assert is_nonconstant_polynomial(2*x**3+x, x)
        assert not is_nonconstant_polynomial(sp.exp(x), x) 
        assert is_nonconstant_polynomial(sp.exp(x), x) is not None
        assert not is_nonconstant_polynomial(x**1.2, x)
        assert not is_nonconstant_polynomial(sp.sqrt(x) + x**2, x )
        
    def test_is_pip(self):
        a = annihilateOp()
        assert not is_pip(sp.Number(1), annihilateOp)
        assert is_pip(a, annihilateOp)
        assert not is_pip(2*a, annihilateOp)
        assert not is_pip(a**2+1, annihilateOp)
        assert is_pip(a**2, a)
        assert is_pip(a**2, annihilateOp)
        assert not is_pip(a**2, createOp)
        assert not is_pip(a**(-1), a)
    
    def has_universal_oper(self):
        expected_to_fail(lambda: has_universal_oper(1))
        
        assert not has_universal_oper(sp.sympify(1))
        assert has_universal_oper(rho)
        assert not has_universal_oper(createOp())
        assert has_universal_oper(rho*createOp())
    
    def test_separate_operator(self):
        op_1 = Operator(1)
        op_2 = Operator(2)
        x = sp.Symbol("x")
        foo = sp.Function("f")(x, op_1)
        
        expected_to_fail(lambda: separate_term_by_nonconstant_polynomiality(x+2))

        assert separate_operator(op_1) == (1, op_1)
        assert separate_operator(x) == (x, 1)
        assert separate_operator(x * op_1) == (x, op_1)
        
        assert separate_operator(x**2) == (x**2, 1)
        assert separate_operator(op_2**2) == (1, op_2**2)
        
        assert (separate_operator(op_1*foo*op_2**2*x**3*2) 
                == (2*x**3, op_1*foo*op_2**2))
        
    def test_separate_term_by_nonconstant_polynomiality(self):
        a_1 = annihilateOp(1)
        ad_2 = createOp(2)
        x = sp.Symbol("x")
        
        foo = separate_term_by_nonconstant_polynomiality
        psvo = PhaseSpaceVariableOperator
        
        expected_to_fail(lambda: foo(a_1+1, psvo))
        
        assert (foo(a_1, psvo)
                == [a_1])
        assert (foo(a_1**2*ad_2, psvo)
                == [1, a_1**2*ad_2])
        assert (foo(a_1**2*ad_2*sp.exp(x), psvo)
                == [sp.exp(x), a_1**2*ad_2])
        assert (foo(a_1*x*sp.exp(ad_2)*ad_2**3 * a_1**0.3 * 2**ad_2, psvo)
                == [x*a_1**1.3*sp.exp(ad_2), ad_2**3, 2**ad_2])
        
    def test_get_factor_nonconstant_polynomiality(self):
        x = sp.Symbol("x")
        foo = get_factor_nonconstant_polynomiality
        
        expected_to_fail(lambda: foo(x+1, x))
        
        assert foo(sp.Number(1), x) == [False]
        assert foo(x, x) == [True]
        assert foo(x*sp.exp(x),x) == [True, False]
        assert foo(3*sp.Symbol("y")*sp.exp(x)*x**4,x) == [False, False, True, False]
        
    def test_collect_alpha_type_oper_from_monomial_by_sub(self):
        sub_cache.clear()
        
        a_1 = annihilateOp(1)
        ad_1 = createOp(1)
        a_2 = annihilateOp(2)
        ad_2 = createOp(2)
        x = sp.Symbol("x")
        
        foo = collect_alpha_type_oper_from_monomial_by_sub
        
        expected_to_fail(lambda : foo(a_1+1, (annihilateOp, createOp)))        
        expected_to_fail(lambda : foo(sp.exp(ad_2), (annihilateOp, createOp)))
        
        assert foo(x)[0] == x
        
        col_ad = foo(ad_1)[1]
        assert isinstance(col_ad, dict)
        assert col_ad[ad_1.sub] == [ad_1, 1]
        
        col_a = foo(a_1)[2]
        assert isinstance(col_a, dict)
        assert col_a[a_1.sub] == [a_1, 1]
        
        expr = 2*x*sp.exp(x) * a_2**2 * a_1 * a_2 * a_1 * ad_2**3 * a_2**1
        non_op, col_ad, col_a = foo(expr)
        
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
        ad = [createOp(i) for i in range(4)]
        x = sp.Symbol("x")
        
        expected_to_fail(lambda: collect_alpha_type_oper_from_monomial_by_sub(a[0]+ad[0]))

        assert foo(a[0]) == [1, a[0]]
        assert (list(sp.ordered(foo(a[0]*a[1]))) 
                == list(sp.ordered([1, a[0], a[1]])))
        assert (list(sp.ordered(foo(a[0]*sp.exp(a[0]*a[1])*a[1]))) 
                == list(sp.ordered([1, a[0]*sp.exp(a[0]*a[1])*a[1]])))
        
        assert (list(sp.ordered(foo(2*x * a[0] * ad[1]*2**a[1] * sp.log(a[2]+a[3]**2)*a[3]**3 * a[1])))
                == list(sp.ordered([2*x, a[0], ad[1]*2**a[1]*a[1], sp.log(a[2]+a[3]**2)*a[3]**3])))
                # May raise an error if sympy's ordering is not its current canon. 