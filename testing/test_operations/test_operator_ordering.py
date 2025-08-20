import pytest
import random
import sympy as sp

from symqups.objects.scalars import Scalar
from symqups.objects.operators import qOp, pOp, annihilateOp, createOp
from symqups.operations.operator_ordering import sOrdering

@pytest.mark.order(3)
class TestOperatorOrdering():
    rand_N = random.randint(1, 20)
    
    def test_sOrdering_construction(self):
        assert sOrdering(1) == 1
        
        a_1 = annihilateOp(1)
        ad_1 = createOp(1)
        a_2 = annihilateOp(2)
        ad_2 = createOp(2)
        x = sp.Symbol("x")
        foo = sp.Function("f")(a_1)
        
        try:
            sOrdering(Scalar())
            raise RuntimeError("Test failed.")
        except:
            pass
        
        assert sOrdering(a_1) == a_1
        assert sOrdering(ad_2) == ad_2
        
        assert not(isinstance(sOrdering(a_2*ad_1), sOrdering)) 
        assert isinstance(sOrdering(a_1*ad_1), sOrdering)
        
        expr = sOrdering(ad_1*a_1)
        assert isinstance(expr, sOrdering) and not(sOrdering(expr).args[0].has(sOrdering))
        
        expr = sOrdering(a_1**3)
        assert isinstance(expr, sp.Pow) and not(expr.has(sOrdering))
        
        expr = sOrdering(a_2*ad_2 + 2*a_2)
        assert isinstance(expr, sp.Add) and expr.has(sOrdering)
        
        assert sOrdering(2*x*ad_1*a_1) == 2*x*sOrdering(ad_1*a_1)
        
        assert isinstance(sOrdering(foo), sOrdering)