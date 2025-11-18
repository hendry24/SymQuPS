import sympy as sp
from symqups._internal.cache import sub_cache

def clear_sub_cache():
    sub_cache.clear()

def arithmetic_test(A):
    assert A+2 == 2+A
    assert 2+A == A+2
    assert A-2 == -2+A
    assert A*2 == 2*A
    assert A/2 == sp.Rational(1,2)*A
    assert A**2 == A*A
    assert A**(0) == 1
    assert sp.sqrt(A)
    
def expected_to_fail(foo : callable):
    try:
        foo()
    except:
        return

    raise RuntimeError("This part is expected to fail, but succeded instead.")