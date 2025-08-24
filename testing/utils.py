import sympy as sp

def arithmetic_test(A):
    assert A+2 == 2+A
    assert 2+A == A+2
    assert A-2 == -2+A
    assert A*2 == 2*A
    assert A/2 == sp.Rational(1,2)*A
    assert A**2 == A*A
    assert A**(0) == 1
    assert sp.sqrt(A)