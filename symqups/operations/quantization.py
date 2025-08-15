import sympy as sp

from ..objects import scalars
from ..objects.cache import _sub_cache
from ..objects.operators import Operator, qOp, pOp, annihilateOp, createOp
from ..utils.multiprocessing import _mp_helper

def s_quantize(expr : sp.Expr):
    '''
    Return the totally-symmetric (Weyl) ordering of the
    input expression containing `Operator'.
    '''
    expr = sp.expand(sp.sympify(expr))
    
    has_qp = expr.has(scalars.q, scalars.p)
    has_alpha = expr.has(scalars.alpha, scalars.alphaD)
    if has_qp and has_alpha:
        msg = "Input expresion contains both (q,p) and (alpha). "
        msg += "Please choose either one. "
        raise TypeError(msg)
    elif has_qp:
        mode = "qp"
    elif has_alpha:
        mode = "alpha"
    else: # nothing to quantize
        return expr 
    
    naive_quantization_dict = {}
    for sub in _sub_cache:
        naive_quantization_dict[scalars.q(sub)] = qOp(sub)
        naive_quantization_dict[scalars.p(sub)] = pOp(sub)
        naive_quantization_dict[scalars.alpha(sub)] = annihilateOp(sub)
        naive_quantization_dict[scalars.alphaD(sub)] = createOp(sub)
    expr_naive_quantized = expr.subs(naive_quantization_dict)
    
    """
    Though 'alpha' is printed first before 'alphaD' in outputs, when 
    we do the sub, the resulting expression is normal-ordered. 
    """
    
    return expr_naive_quantized

class WeylTransform():
    
    def __new__(cls, expr : sp.Expr):
        expr = sp.sympify(expr)
        expr = expr.expand()
        
        _screen_type(expr, Operator, "WeylTransform")

        if not(expr.has(scalars.q, scalars.p, scalars.W)):
            return expr
       
        if isinstance(expr, (scalars.q, scalars.p, scalars.WignerFunction)):
            return expr.weyl_transform()
        
        if isinstance(expr, sp.Add):
            return _mp_helper(expr.args, WeylTransform)
        
        if isinstance(expr, sp.Pow):
            base : scalars.q | scalars.p = expr.args[0]
            exponent = expr.args[1]
            return base.weyl_transform() ** exponent
        
        if isinstance(expr, sp.Mul):
            return
        
        _invalid_input()