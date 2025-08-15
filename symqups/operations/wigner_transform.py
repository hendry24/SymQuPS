import sympy as sp

from ..objects.operators import Operator
from .star_product import Star
from ..utils.multiprocessing import _mp_helper

class WignerTransform():
    """
    The Wigner transform.
    
    Parameters
    ----------
    
    A : sp.Expr
    
    """

    def __new__(cls, A : sp.Expr):

        A = sp.expand(sp.sympify(A))
        
        if not(A.has(Operator)):
            return A

        if isinstance(A, Operator):
            return A.wigner_transform()
                        
        if isinstance(A, (sp.Add, sp.Mul)):
            res = _mp_helper(A.args, WignerTransform)
            if isinstance(A, sp.Add):
                return sp.Add(*res)
            return Star(*res).expand()
        
        if isinstance(A, sp.Pow):
            base : Operator = A.args[0]
            exponent = A.args[1]
            return (base.wigner_transform() ** exponent).expand()
        
        raise ValueError(r"Invalid input in WignerTransform: {%s}" %
                         (sp.latex(A)))