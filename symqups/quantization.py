import sympy as sp

from .objects.operators import Operator

from ._internal.grouping import qpType, PhaseSpaceObject
from ._internal.basic_routines import deep_screen_type, operation_routine

from .manipulations import qp2alpha, sc2op

from .ordering import sOrdering

def _prepare_for_quantization(expr : sp.Expr) -> sp.Expr:
    expr = sp.sympify(expr)
    
    deep_screen_type(expr, Operator, "_prepare_for_quantization")
    
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
    '''
    Return the totally-symmetric (Weyl) ordering of the
    input expression, most generally a polynomial in 
    (q,p) or (alpha, alphaD).
    '''
    expr = _prepare_for_quantization(expr)
    return operation_routine(expr,
                             "s_quantize",
                             (),
                             (Operator),
                             {PhaseSpaceObject : expr},
                             {(sp.Add, sp.Mul, sp.Pow, sp.Function, PhaseSpaceObject) 
                              : sOrdering(naive_quantize(expr))}
                             )