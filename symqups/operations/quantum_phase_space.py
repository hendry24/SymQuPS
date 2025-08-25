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
        
class CGTransform(sp.Expr, UnBoppable, NotAnOperator):
    def __new__(cls, expr : sp.Expr, *_vars):
        """
        oper -> quantum ps vars
        """
        
        def treat_add(A : sp.Expr) -> sp.Expr:
            return sp.Add(*_mp_helper(A.args, CGTransform))
        
        def treat_substitutable(A : sp.Expr) -> sp.Expr:
            return A.subs({op(sub) : sc(sub) for sub in sub_cache for op,sc in [[annihilateOp, alpha],
                                                                                [createOp, alphaD]]})
        
        def treat_function(A : sp.Function) -> sp.Expr:
            sOrdering_of_A = sOrdering(A)
            if not(isinstance(sOrdering_of_A, sOrdering)):
                return treat_substitutable(sOrdering_of_A)
            return CGTransform(sOrdering_of_A)
        
        def treat_sOrdering(A : sOrdering, vars = None) -> sp.Expr:
            if A.args[1] != s.val:
                return CGTransform(A.express(s.val))
            
            if not(vars):
                vars = [sc(atom.sub) for sc,op in [[alpha, annihilateOp], 
                                                   [alphaD, createOp]] 
                        for atom in A.atoms(op)]
            
            return super(CGTransform, cls).__new__(cls, A, *vars)
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
            return Star(*_mp_helper(A.args, CGTransform))
            
        expr = qp2a(sp.sympify(expr))
        return operation_routine(expr,
                                "CG_transform",
                                [],
                                [],
                                {Operator : expr},
                                {sp.Add : treat_add,
                                sp.Mul : treat_mul,
                                (Operator, sp.Pow) : treat_substitutable,
                                sp.Function : treat_function,
                                sOrdering : lambda A: treat_sOrdering(A, _vars)})
        
    def _latex(self, printer):
        return r"\mathcal{G}\left[{%s}\right]" % sp.latex(self.args[0])