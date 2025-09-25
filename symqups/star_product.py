import sympy as sp
import functools

from ._internal.basic_routines import operation_routine
from ._internal.grouping import UnBoppable, PhaseSpaceVariable, Defined, qpType, HilbertSpaceObject, alphaType
from ._internal.cache import sub_cache
from ._internal.multiprocessing import mp_helper
from ._internal.preprocessing import preprocess_class

from .objects.base import Base
from .objects.scalars import alpha, alphaD, StateFunction

from .manipulations import qp2alpha

from . import s as CahillGlauberS

###

class _Primed(Base):
    def _get_symbol_name_and_assumptions(cls, psv):
        return r"{%s}'" % sp.latex(psv), {"commutative" : False} 
    
    def __new__(cls, psv : PhaseSpaceVariable):

        def make(A : alphaType):
            return super(_Primed, cls).__new__(cls, A)
        
        def prime_expr(A : sp.Expr) -> sp.Expr:
            subs_dict = {x : make(x) for sub in sub_cache for x in (alpha(sub), alphaD(sub))}
            return A.xreplace(subs_dict)
        
        return operation_routine(psv,
                                 _Primed,
                                 [],
                                 [],
                                 {_Primed : prime_expr},
                                 {PhaseSpaceVariable : make})
        
def _deprime(expr : sp.Expr) -> sp.Expr:
    subs_dict = {_Primed(x) : x for sub in sub_cache for x in (alpha(sub), alphaD(sub))}
    return expr.xreplace(subs_dict)

class _CannotBoppFlag(TypeError):
    pass

def _star_Bopp_monomial_A_times_B(A : sp.Expr, B : sp.Expr, sgn : int) -> sp.Expr:
    # We put this here to make it pickleable
    
    s = CahillGlauberS.val
    
    if A.is_Mul:
        args = A.args
    else:
        args = [A]
        
    non_psv = sp.Integer(1)
    out = _Primed(B)
    
    for arg in args:        
        if arg.has(alphaD):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
                
            sub = list(arg.atoms(alphaD))[0].sub
            a, ad = alpha(sub), alphaD(sub)
            
            for _ in range(n):
                out = ad*out + sp.Rational(1,2)*(s-sgn)*sp.Derivative(out, _Primed(a))
            
        elif arg.has(alpha):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
            
            sub = list(arg.atoms(alpha))[0].sub
            a, ad = alpha(sub), alphaD(sub)
            
            for _ in range(n):
                out = a*out + sp.Rational(1,2)*(s+sgn)*sp.Derivative(out, _Primed(ad))
                
        else:
            non_psv *= arg
            
    return _deprime((non_psv*out).doit().expand())
    
def _star_base(f : sp.Expr, g : sp.Expr) -> sp.Expr:
        
    if not(f.has(PhaseSpaceVariable)) or not(g.has(PhaseSpaceVariable)):
        return f*g
    
    if f.has(UnBoppable) and g.has(UnBoppable):
        raise _CannotBoppFlag
    
    if f.is_polynomial(alpha, alphaD):
        to_Bopp = f
        sgn = 1
        other = g
    elif g.is_polynomial(alpha, alphaD):
        to_Bopp = g
        sgn = -1
        other = f
    else:
        raise _CannotBoppFlag
    
    def treat_add(A : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, functools.partial(_star_Bopp_monomial_A_times_B, 
                                                           B=other, 
                                                           sgn=sgn)))
    
    return operation_routine(to_Bopp,
                             Star,
                             [],
                             [HilbertSpaceObject],
                             {},
                             {sp.Add : treat_add,
                              (sp.Mul, 
                               sp.Pow, 
                               PhaseSpaceVariable,
                               StateFunction) 
                                : _star_Bopp_monomial_A_times_B(to_Bopp, other, sgn)}
                            )

@preprocess_class
class Star(sp.Expr, UnBoppable, Defined):
    """
    The s-parameterized star-product `A(q,p) ★ B(q,p) ★ ...` (or the `alpha` equivalent), 
    calculated using the Bopp shift.

    Parameters
    ----------

    *args
        The factors of the star-product, ordered from first to last. Since the algorithm
        utilizes the Bopp shift, only one operand can be a non-polynomial.

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p
    
    See Also
    --------
    
    .Bopp : Bopp shift the input expression. 
    
    """

    @staticmethod
    def _definition():
        out = r"f\left(\bm{\alpha},\overline{\bm{\alpha}}\right)"
        out += r"\mathbin{\star_s}"
        out += r"g\left(\bm{\alpha},\overline{\bm{\alpha}}\right)"
        out += r"= f\left(\bm{\alpha}, \overline{\bm{\alpha}}\right)"
        out += r"\exp\left( \frac{s+1}{2} \overset{\leftarrow}{\partial_{\bm{\alpha}}}"
        out += r"\overset{\rightarrow}{\partial_{\overline{\bm{\alpha}}}}"
        out += r"+ \frac{s-1}{2} \overset{\leftarrow}{\partial_{\overline{\bm{\alpha}}}}"
        out += r"\overset{\rightarrow}{\partial_{\bm{\alpha}}} \right)"
        out += r"g \left(\bm{\alpha}, \overline{\bm{\alpha}} \right)"
        return sp.Symbol(out)
    definition = _definition()
    
    def __new__(cls, *args : sp.Expr) -> sp.Expr:
        if not(args):
            return sp.Integer(1)
        
        unevaluated_args = []
        
        out = sp.Integer(1)
        for k, arg in enumerate(args):
            try:
                if arg.has(qpType):
                    arg = qp2alpha(arg)
                out = _star_base(out, arg)
            except _CannotBoppFlag:
                if out != 1:
                    unevaluated_args.append(out)
                out = arg
                if k == (len(args)-1):
                    unevaluated_args.append(arg)
        
        if unevaluated_args:
            return super().__new__(cls, *unevaluated_args)
        
        return out
        
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\star_s \left({%s}\right)" % sp.latex(arg)
        return out