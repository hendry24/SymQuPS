import pytest
import sympy as sp

from symqups.objects.scalars import Scalar, W
from symqups.objects.operators import Operator
from symqups._internal.multiprocessing import mp_helper, MP_CONFIG

from symqups._internal.cache import sub_cache
sub_cache.clear()

def mp_helper_foo(x):
        return x+2

@pytest.mark.fast
def testmp_helper():
    inpt = [1, sp.Symbol("x"), Scalar(),
            Operator(), W]
    
    global MP_CONFIG
    enable_default = MP_CONFIG["enable"]
    MP_CONFIG["min_num_args"] = 0

    for enable in [True, False]:
        MP_CONFIG["enable"] = enable
        assert (mp_helper(inpt, mp_helper_foo) 
                == list(map(mp_helper_foo, inpt)))
    
    MP_CONFIG["enable"] = enable_default