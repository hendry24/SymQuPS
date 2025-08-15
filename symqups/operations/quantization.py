import sympy as sp

from ..objects import scalars
from ..objects.operators import Operator, qOp, pOp, rho
from ..utils.multiprocessing import _mp_helper

def s_quantize(expr : sp.Expr):
    '''
    Return the totally-symmetric (Weyl) ordering of the
    input expression containing `Operator'.
    '''
    expr = sp.expand(sp.sympify(expr))
    
    _screen_type(expr, (scalars.Scalar), "weyl_ordering")
    
    if not(expr.has(qOp, pOp)):
        return expr
    
    if isinstance(expr, (qOp, pOp, sp.Pow)):
        return expr
    
    if isinstance(expr, sp.Add):
        return _mp_helper(expr.args, weyl_ordering)
        
    _invalid_input(expr, "weyl_ordering")

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