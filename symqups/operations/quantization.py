import sympy as sp

from ..objects.scalars import alpha, alphaD
from ..objects.operators import Operator, annihilateOp, createOp

from .._internal.grouping import qpType, PhaseSpaceObject
from .._internal.cache import sub_cache
from .._internal.basic_routines import deep_screen_type, operation_routine

from ..utils.algebra import qp2a

from .operator_ordering import sOrdering

def _prepare_for_quantization(expr : sp.Expr) -> sp.Expr:
    expr = sp.sympify(expr)
    
    deep_screen_type(expr, Operator, "_prepare_for_quantization")
    
    if expr.has(qpType):
        expr = qp2a(expr)
    
    return sp.expand(expr)

def naive_quantize(expr : sp.Expr) -> sp.Expr:
    expr = _prepare_for_quantization(expr)
    
    naive_quantization_dict = {}
    for sub in sub_cache:
        naive_quantization_dict[alpha(sub)] = annihilateOp(sub)
        naive_quantization_dict[alphaD(sub)] = createOp(sub)
    expr_naive_quantized = expr.subs(naive_quantization_dict)

    """
    Though 'alpha' is printed first before 'alphaD' in outputs, when 
    we do the sub, the resulting expression is normal-ordered, corresponding
    to s = 1. This applies in the multipartite case: all `alphaD` are 
    situated to the left of all `alpha`.
    """
    
    return sp.expand(expr_naive_quantized)

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