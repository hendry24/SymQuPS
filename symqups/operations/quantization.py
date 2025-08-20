import sympy as sp

from .. import s
from ..objects.base import qpTypePSO, alphaTypePSO, PhaseSpaceObject
from ..objects.scalars import alpha, alphaD
from ..objects.cache import _sub_cache
from ..objects.operators import Operator, annihilateOp, createOp
from ..utils._internal._basic_routines import _operation_routine, _invalid_input
from ..utils.multiprocessing import _mp_helper
from ..utils.algebra import qp2a
from .operator_ordering import s_ordered

def _prepare_for_quantization(expr : sp.Expr) -> sp.Expr:
    expr = sp.expand(sp.sympify(expr))
    
    if expr.has(Operator):
        raise TypeError("Input can not be Operator objects.")
    
    if expr.has(qpTypePSO):
        expr = qp2a(expr)
    
    return sp.expand(expr)

def naive_quantize(expr : sp.Expr):
    expr = _prepare_for_quantization(expr)
    
    naive_quantization_dict = {}
    for sub in _sub_cache:
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

def s_quantize(expr : sp.Expr):
    '''
    Return the totally-symmetric (Weyl) ordering of the
    input expression, most generally a polynomial in 
    (q,p) or (alpha, alphaD).
    '''
    
    def do_monomial(A):
        out = 1
        
        if isinstance(A, sp.Mul):
            decompose_A = A.args
        elif isinstance(A, (sp.Pow, alphaTypePSO)):
            decompose_A = [A]
        else:
            raise TypeError(f"Invalid type encountered: {type(A).__name__}")
        
        exp_ad = {sub : 0 for sub in _sub_cache}
        exp_a = {sub : 0 for sub in _sub_cache}
        for A_ in decompose_A:
            if not(A_.has(PhaseSpaceObject)):
                out *= A_
            elif isinstance(A_, alphaD):
                exp_ad[A_.sub] = 1
            elif A_.has(alphaD):
                exp_ad[A_.args[0].sub] = A_.args[1]
            elif isinstance(A_, alpha):
                exp_a[A_.sub] = 1
            elif A_.has(alpha):
                exp_a[A_.args[0].sub] = A_.args[1]
            else:
                _invalid_input(A, "do_monomial' in 's_quantize")
        
        for sub in _sub_cache:
            out *= s_ordered(sub, exp_ad[sub], exp_a[sub])
            
        return sp.expand(out)
    
    expr = _prepare_for_quantization(expr)
    return _operation_routine(expr,
                              "s_quantize",
                              (Operator),
                              (PhaseSpaceObject,),
                              expr,
                              (
                                  (sp.Add,),
                                  lambda A: sp.Add(*_mp_helper(A.args, s_quantize))
                              ),
                              (
                                  (sp.Mul, sp.Pow, PhaseSpaceObject),
                                  do_monomial
                              )
    )