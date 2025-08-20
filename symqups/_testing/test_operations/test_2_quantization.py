import pytest
import sympy as sp

from symqups.objects.base import qpTypePSO
from symqups.objects.scalars import q, p, alpha, alphaD
from symqups.objects.operators import qOp, pOp, annihilateOp, createOp
from symqups.operations.quantization import _prepare_for_quantization, naive_quantize, s_quantize

@pytest.mark.order(3)
class TestQuantization():
    
    def test_naive_quantize(self):
        pass