import sympy as sp
import functools

from ._internal.basic_routines import operation_routine
from ._internal.grouping import UnOperBoppable, Defined, qpType, HilbertSpaceObject, PhaseSpaceVariableOperator, PhaseSpaceObject, alphaType
from ._internal.cache import sub_cache
from ._internal.multiprocessing import mp_helper
from ._internal.preprocessing import preprocess_class

from .objects.base import Base
from .objects.operators import annihilateOp, createOp, densityOp

from .manipulations import qp2alpha

from . import s as CahillGlauberS

###

class _Tilded(Base):
    def _get_symbol_name_and_assumptions(cls, psvOp):
        return r"\widetilde{%s}" %  sp.latex(psvOp), {"commutative" : False}
    
    def __new__(cls, psvOp : PhaseSpaceVariableOperator):
        
        def make(A : alphaType):
            return super(_Tilded, cls).__new__(cls, A)
        
        def tilde_expr(A : sp.Expr) -> sp.Expr:
            subs_dict = {x : make(x) for sub in sub_cache for x in (createOp(sub),
                                                                    annihilateOp(sub))}
            return A.xreplace(subs_dict)
        
        return operation_routine(psvOp,
                                 _Tilded,
                                 [],
                                 [],
                                 {_Tilded : tilde_expr},
                                 {PhaseSpaceVariableOperator : make})
        
def _detilde(expr : sp.Expr) -> sp.Expr:
    subs_dict = {_Tilded(x) : x for sub in sub_cache
                 for x in (createOp(sub),
                           annihilateOp(sub))}
    return expr.xreplace(subs_dict)

class _CannotOperBoppFlag(TypeError):
    pass

def _hatted_star_OperBopp_monomial_A_times_B(A : sp.Expr, B : sp.Expr) -> sp.Expr:
    """
    No need for `sgn` since the hatted star product is commutative and we can always
    swap the order of operation. 
    """
    
    s = CahillGlauberS.val
    
    if A.is_Mul:
        args = A.args
    else:
        args = [A]
        
    non_psvOp = sp.Integer(1)
    out = _Tilded(B)
    
    for arg in reversed(args):
        if arg.has(createOp):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
            
            sub = list(arg.atoms(createOp))[0].sub
            a, ad = annihilateOp(sub), createOp(sub)
            
            for _ in range(n):
                out = ad*out - sp.Rational(1,2)*(s-1)*sp.Derivative(out, _Tilded(a))
        
        elif arg.has(annihilateOp):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
            
            sub = list(arg.atoms(annihilateOp))[0].sub
            a, ad = annihilateOp(sub), createOp(sub)
            
            for _ in range(n):
                out = a*out - sp.Rational(1,2)*(s+1)*sp.Derivative(out, _Tilded(ad))
        
        else:
            non_psvOp *= arg
            
    return _detilde((non_psvOp*out).doit().expand())
    
def _hatted_star_base(F : sp.Expr, G : sp.Expr) -> sp.Expr:
    
    if not(F.has(PhaseSpaceVariableOperator, densityOp) 
           and G.has(PhaseSpaceVariableOperator, densityOp)):
        return F*G
    
    if F.has(UnOperBoppable) and G.has(UnOperBoppable):
        raise _CannotOperBoppFlag
    
    if F.is_polynomial(annihilateOp, createOp):
        to_Bopp = F
        other = G
    elif G.is_polynomial(annihilateOp, createOp):
        to_Bopp = G
        other = F
    else:
        raise _CannotOperBoppFlag
    
    def treat_add(to_Bopp : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(to_Bopp.args, 
                                 functools.partial(_hatted_star_OperBopp_monomial_A_times_B, 
                                                   B = other))
                      )

    return operation_routine(to_Bopp,
                             HattedStar,
                             [],
                             [PhaseSpaceObject],
                             {},
                             {sp.Add : treat_add,
                             (sp.Mul,
                              sp.Pow,
                              PhaseSpaceVariableOperator,
                              densityOp) 
                             : _hatted_star_OperBopp_monomial_A_times_B(to_Bopp, other)
                             }
                             )

@preprocess_class
class HattedStar(sp.Expr, UnOperBoppable, HilbertSpaceObject, Defined):
    @staticmethod
    def _definition():
        out = r"f\left(\hat{\bm{a}},\hat{\bm{a}}^\dagger\right)"
        out += r"\mathbin{\widetilde{\star}_s}"
        out += r"g\left(\hat{\bm{a}},\hat{\bm{a}}^\dagger\right)"
        out += r"= f\left(\hat{\bm{a}}-\frac{1+s}{2}\left[\hat{\bm{a}},\cdot\right],"
        out += r"\hat{\bm{a}}^\dagger - \frac{1-s}{2}\left[\hat{\bm{a}}^\dagger, \cdot\right]\right)"
        out += r"g\left(\hat{\bm{a}},\hat{\bm{a}}^\dagger\right)"
        return sp.Symbol(out)
    definition = _definition()
    
    def __new__(cls, *args) -> sp.Expr:
        if not(args):
            return sp.Integer(1)
        
        # NOTE: hStar of two polynomials equals the s-ordering of their product,
        # since the iCGTransform of a product of two polynomials is that product,
        # converted into operators, and s-ordered. This helps shortcut the evaluation.
        
        UnOperBoppable_args = []
        
        out = sp.Integer(1)
        for k,arg in enumerate(args):
            try:
                if arg.has(qpType):
                    arg = qp2alpha(arg)
                out = _hatted_star_base(out, arg)
            except _CannotOperBoppFlag:
                if out != 1:
                    UnOperBoppable_args.append(out)
                out = arg
                if k == (len(args)-1):
                    UnOperBoppable_args.append(arg)
                    
        if UnOperBoppable_args:
            return super().__new__(cls, *UnOperBoppable_args)
    
        return out
    
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\widetilde{\star}_s \left({%s}\right)" % sp.latex(arg)
        return out