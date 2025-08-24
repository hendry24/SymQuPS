import pytest
import sympy as sp
import random

from symqups.objects.scalars import Scalar, q, p, alpha, alphaD, W, hbar, mu
from symqups.objects.operators import Operator, qOp, pOp, createOp, annihilateOp
from symqups.utils.multiprocessing import _mp_helper, MP_CONFIG
from symqups.utils.algebra import get_random_poly, define, qp2a

def mp_helper_foo(x):
        return x+2

@pytest.mark.fast
def test_mp_helper():
    inpt = [1, sp.Symbol("x"), Scalar(),
            Operator(), W]
    
    global MP_CONFIG
    enable_default = MP_CONFIG["enable"]
    MP_CONFIG["min_num_args"] = 0

    for enable in [True, False]:
        MP_CONFIG["enable"] = enable
        assert (_mp_helper(inpt, mp_helper_foo) 
                == list(map(mp_helper_foo, inpt)))
    
    MP_CONFIG["enable"] = enable_default