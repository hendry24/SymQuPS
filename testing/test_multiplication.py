import pytest
import sympy as sp

from symqups.objects.operators import Operator
from symqups.objects.scalars import Scalar
from symqups import sMul

@pytest.mark.fast
@pytest.mark.order(0)
def test_Mul():
    sc_1 = Scalar(1)
    sc_2 = Scalar(2)
    op_1 = Operator(1)
    op_2 = Operator(2)
    x = sp.Symbol("x")
    
    assert 1*x == x
    assert x*1 == x
    assert 1*sc_1 == sc_1
    assert sc_1*1 == sc_1
    assert 1*op_1 == op_1
    assert op_1*1 == op_1
    
    # We cannot directly check whether the value represented
    # by sMul is equal to that of Mul
    
    assert sc_1*sc_2 == sMul(sc_1, sc_2)
    assert op_1*op_2 == sMul(sp.ordered([op_1, op_2]))