import pytest
import sympy as sp

from symqups.objects.operators import qOp, pOp, annihilateOp, createOp, densityOp
from symqups.operations.dagger import dagger
from symqups.utils.algebra import get_random_poly

@pytest.mark.fast
@pytest.mark.order(6)
def test_dagger():
    for herm_op in [qOp(), pOp(), densityOp()]:
        assert dagger(herm_op) == herm_op

    assert dagger(annihilateOp()) == createOp()
    assert (dagger(annihilateOp().define())-createOp().define()).expand() == 0
    assert dagger(createOp()) == annihilateOp()            
    assert (dagger(createOp().define())-annihilateOp().define()).expand() == 0

    rand_poly = get_random_poly(objects = (1, sp.Symbol("x"), qOp(), annihilateOp(),
                                            createOp(), annihilateOp()),
                                coeffs = list(range(10)) + sp.symbols([]))
    assert (dagger(dagger(rand_poly)) - rand_poly).expand() == 0