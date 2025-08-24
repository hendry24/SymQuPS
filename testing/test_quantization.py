import pytest
import sympy as sp

from symqups import s
from symqups.objects.scalars import q, p, alpha, alphaD, W
from symqups.objects.operators import Operator, qOp, pOp, annihilateOp, createOp
from symqups.utils.algebra import get_random_poly, qp2a
from symqups.operations.operator_ordering import sOrdering

from symqups.operations.quantization import s_quantize

@pytest.mark.fast
def test_s_quantize():
    # Since 's_quantize' depends heavily on 'naive_quantize',
    # this test also serves to test 'naive_quantize'
    qq = q()
    q_op = qOp()
    pp = p()
    p_op = pOp()
    a = alpha()
    a_op = annihilateOp()
    ad = alphaD()
    ad_op = createOp()
    x = sp.Symbol("x")
    
    for obj in (s, W, Operator()):
        try:
            s_quantize(obj)
            raise RuntimeError("Test failed.")
        except:
            pass
        
    assert s_quantize(1) == 1
    
    assert s_quantize(x) == x
    assert s_quantize(x+2) == (x+2)
    assert s_quantize(sp.exp(x)) == sp.exp(x)

    assert s_quantize(a) == a_op
    assert s_quantize(qq) == qp2a(q_op)
    
    assert s_quantize(a+ad) == (a_op+ad_op)
    assert s_quantize(2*ad) == 2*ad_op
    assert s_quantize(ad**2) == ad_op**2
    assert s_quantize(sp.exp(2*a)) == sp.exp(2*a_op)
    
    assert s_quantize(sp.exp(a*ad)) == sOrdering(sp.exp(ad_op*a_op))
                                        # NOTE: sympy orders alphaD() before alpha()
                                        # so when it is naive-quantized, the resulting
                                        # expression is normal-ordered.