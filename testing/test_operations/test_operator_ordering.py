import pytest
import sympy as sp

from symqups.objects.scalars import Scalar
from symqups.objects.operators import qOp, pOp, annihilateOp, createOp
from symqups.operations.operator_ordering import sOrdering

@pytest.mark.order(3)
class TestOperatorOrdering():
    
    def test_sOrdering(self):
        a_1 = annihilateOp(1)
        pass