import sympy as sp

from ._internal.multiprocessing import mp_helper
from ._internal.basic_routines import operation_routine
from ._internal.grouping import PhaseSpaceVariable

from .objects.operators import Operator

from .star_product import Star
from .ordering import sOrdering
from .manipulations import qp2a, sc2op, op2sc

from . import s as CahillGlauberS

###
        
def CG_transform(expr : sp.Expr) -> sp.Expr:
    """
    oper -> quantum ps vars
    """
    
    def treat_add(A : sp.Expr) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, CG_transform))
    
    def treat_substitutable(A : sp.Expr) -> sp.Expr:
        return op2sc(A)
    
    def treat_function(A : sp.Function) -> sp.Expr:
        sOrdering_of_A = sOrdering(A)
        if not(isinstance(sOrdering_of_A, sOrdering)):
            return treat_substitutable(sOrdering_of_A)
        return CG_transform(sOrdering_of_A)
    
    def treat_sOrdering(A : sOrdering) -> sp.Expr:
        if (A.args[1] != CahillGlauberS.val):
            return CG_transform(A.express(CahillGlauberS.val))
        
        # In the following A has the same s-value as the transform,
        # so we can simply discard the braces and replace the operators
        # by the corresponding phase-space variables, since the CG transform
        # of an s-ordered operator is a straightforward replacement.
        return treat_substitutable(A.args[0])
                    
    def treat_mul(A : sp.Expr) -> sp.Expr:
        return Star(*mp_helper(A.args, CG_transform))
        
    expr = qp2a(sp.sympify(expr))
    return operation_routine(expr,
                            "CG_transform",
                            [],
                            [PhaseSpaceVariable],
                            {Operator : expr},
                            {sp.Add : treat_add,
                            sp.Mul : treat_mul,
                            (Operator, sp.Pow) : treat_substitutable,
                            sp.Function : treat_function,
                            sOrdering : treat_sOrdering})
    
def iCG_transform(expr : sp.Expr, lazy=False) -> sp.Expr:   
    return sOrdering(sc2op(expr), lazy=lazy)