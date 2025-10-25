import sympy as sp

from ._internal.grouping import HilbertSpaceObject
from ._internal.basic_routines import deep_screen_type
from ._internal.preprocessing import preprocess_func

from .manipulations import sc2op

from .cg import iCGTransform

@preprocess_func
def naive_quantize(expr : sp.Expr) -> sp.Expr:
    # Though 'alpha' is printed first before 'alphaD' in outputs, when 
    # we do the sub, the resulting expression is normal-ordered, corresponding
    # to s = 1. This applies in the multipartite case: all `alphaD` are 
    # situated to the left of all `alpha`.
    deep_screen_type(expr, HilbertSpaceObject, naive_quantize)
    return sc2op(expr)

def s_quantize(expr : sp.Expr) -> sp.Expr:
    return iCGTransform(expr)