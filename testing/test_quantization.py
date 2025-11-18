import pytest
import sympy as sp

from symqups.objects.scalars import alpha, alphaD
from symqups.objects.operators import annihilateOp, createOp
from symqups.ordering import normal_order, Weyl_order, antinormal_order
from symqups.quantization import normal_quantize, Weyl_quantize, antinormal_quantize
from symqups.manipulations import normal_ordered_equivalent

from symqups._internal.cache import sub_cache
sub_cache.clear()
###

@pytest.mark.fast
def test_quantization():
    a, ad  = alpha(), alphaD()
    aOp, adOp = annihilateOp(), createOp()
    for q, o in zip([normal_quantize, Weyl_quantize, antinormal_quantize],
                    [normal_order, Weyl_order, antinormal_order]):
        assert sp.expand(normal_ordered_equivalent(q(a*ad) - o(aOp*adOp))) == 0