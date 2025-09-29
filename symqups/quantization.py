import sympy as sp

from ._internal.grouping import qpType, PhaseSpaceObject, HilbertSpaceObject
from ._internal.basic_routines import deep_screen_type, operation_routine
from ._internal.preprocessing import preprocess_func

from .manipulations import qp2alpha, sc2op

from .ordering import sOrdering

@preprocess_func
def _prepare_for_quantization(expr : sp.Expr) -> sp.Expr:
    
    deep_screen_type(expr, HilbertSpaceObject, "_prepare_for_quantization")
    
    if expr.has(qpType):
        expr = qp2alpha(expr)
    
    return sp.expand(expr)

def naive_quantize(expr : sp.Expr) -> sp.Expr:
    # Though 'alpha' is printed first before 'alphaD' in outputs, when 
    # we do the sub, the resulting expression is normal-ordered, corresponding
    # to s = 1. This applies in the multipartite case: all `alphaD` are 
    # situated to the left of all `alpha`.
    return sc2op(_prepare_for_quantization(expr))

def s_quantize(expr : sp.Expr) -> sp.Expr:
    expr = _prepare_for_quantization(expr)
    return operation_routine(expr,
                             s_quantize,
                             (),
                             (HilbertSpaceObject),
                             {PhaseSpaceObject : expr},
                             {(sp.Add, sp.Mul, sp.Pow, sp.Function, PhaseSpaceObject) 
                              : sOrdering(naive_quantize(expr))}
                             )