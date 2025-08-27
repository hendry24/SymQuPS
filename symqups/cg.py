import sympy as sp

from ._internal.multiprocessing import mp_helper
from ._internal.basic_routines import operation_routine
from ._internal.operator_handling import collect_alpha_type_oper_from_monomial_by_sub
from ._internal.grouping import PhaseSpaceVariable, PhaseSpaceObject
from ._internal.cache import sub_cache

from .objects.scalars import W, StateFunction
from .objects.operators import Operator, densityOp, rho, qOp, pOp

from .star_product import Star
from .ordering import sOrdering
from .manipulations import qp2alpha, op2sc, alpha2qp
from .utils import get_N

from . import s as CahillGlauberS
from . import pi, hbar

###

class CGTransform(sp.Expr, PhaseSpaceObject):
    def __new__(cls, expr : sp.Expr):
        """
        oper -> quantum ps vars
        """
        
        def treat_add(A : sp.Expr) -> sp.Expr:
            return sp.Add(*mp_helper(A.args, CGTransform))
        
        def treat_substitutable(A : sp.Expr) -> sp.Expr:
            if isinstance(A, densityOp):
                return (2*pi.val*hbar.val)**get_N() * W
            return op2sc(A)
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
            return Star(*mp_helper(A.args, CGTransform))
            
        def treat_sOrdering(A : sOrdering) -> sp.Expr:
            if (A.args[1] != CahillGlauberS.val):
                if not(A.args[0].is_polynomial(Operator)):
                    return make(A)
                return CGTransform(A.express(CahillGlauberS.val))
            
            # In the following A has the same s-value as the transform,
            # so we can simply discard the braces and replace the operators
            # by the corresponding phase-space variables, since the CG transform
            # of an s-ordered operator is a straightforward replacement.
            return treat_substitutable(A.args[0])
        
        def treat_function(A : sp.Function):
            opers_in_A = A.atoms(Operator)
            
            if rho in opers_in_A:
                return make(A)
            
            evaluable = True
            _, col_ad, col_a = collect_alpha_type_oper_from_monomial_by_sub(sp.Mul(*opers_in_A))
            for sub in sub_cache:
                if col_ad[sub][1] > 0 and col_a[sub][1] > 0:
                    evaluable = False
                    break
            
            ###
                 
            # NOTE: f(qOp) and g(pOp) are also evaluable for the special case where s=0.
            # Compared to above, the following block is "lazier" in the sense that it only
            # allows functions with atomic arguments, e.g. f(qOp) but not f(zeta*qOp).
            if not(evaluable) and CahillGlauberS.val==0:
                sub_found = {}
                evaluable = True
                for arg in set(alpha2qp(A).args):
                    if isinstance(arg, (qOp, pOp)):
                        if sub in sub_found: # same sub twice means that qOp and pOp are both present.
                            evaluable = False
                            break
                        sub_found.append(arg.sub)
                    else: 
                        evaluable = False
                        break
                
            ###
            
            if evaluable:
                return treat_substitutable(A)
            return make(A)
            
        def make(A : sp.Expr):
            return super(CGTransform, cls).__new__(cls, A)
            
        expr = qp2alpha(sp.sympify(expr))
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
        
    def _latex(self, printer):
        return r"\mathcal{W}_{s={%s}}\left[{%s}\right]" % (sp.latex(CahillGlauberS.val),
                                                           sp.latex(self.args[0]))
    
###
    
def iCG_transform(expr : sp.Expr, lazy=False) -> sp.Expr:
    def treat_add(A : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, iCG_transform))
    
    def treat_der(A : sp.Derivative) -> sp.Expr:
        pass
    
    def treat_pow(A : sp.Pow) -> sp.Expr:
        pass
    
    return operation_routine(expr, 
                             "iCG_transform",
                             [],
                             [Operator],
                             {(PhaseSpaceVariable, StateFunction) : expr},
                             {sp.Add : treat_add})