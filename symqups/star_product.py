import sympy as sp
import functools

from ._internal.basic_routines import operation_routine
from ._internal.grouping import UnBoppable, UnDBoppable, PhaseSpaceVariable, Defined, qpType, HilbertSpaceObject, PhaseSpaceVariableOperator, PhaseSpaceObject
from ._internal.cache import sub_cache
from ._internal.multiprocessing import mp_helper

from .objects.base import Base
from .objects.scalars import alpha, alphaD
from .objects.operators import annihilateOp, createOp

from .manipulations import qp2alpha, Commutator

from . import s as CahillGlauberS

###

# Star product
##############

class _Primed(Base):
    def _get_symbol_name_and_assumptions(cls, psv):
        return r"{%s}'" % sp.latex(psv), {"commutative" : False} 
    
    def __new__(cls, psv : PhaseSpaceVariable):

        def make(A : PhaseSpaceVariable):
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
        
def _deprime(expr : sp.Expr):
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
            
    return non_psv*out
    
def _star_base(f : sp.Expr, g : sp.Expr) -> sp.Expr:
        
    if not(f.has(PhaseSpaceVariable) and g.has(PhaseSpaceVariable)):
        return f*g
    
    if f.is_polynomial(alpha, alphaD):
        to_Bopp = f
        sgn = 1
        other = g
    elif g.is_polynomial(alpha, alphaD):
        to_Bopp = f
        sgn = -1
        other = g
    else:
        raise _CannotBoppFlag
    
    def treat_add(A : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, functools.partial(_star_Bopp_monomial_A_times_B, 
                                                           B=other, 
                                                           sgn=sgn)))
    
    primed_out = operation_routine(to_Bopp,
                                   Star,
                                   [],
                                   [HilbertSpaceObject],
                                   {},
                                   {sp.Add : treat_add,
                                    (sp.Mul, 
                                     sp.Pow, 
                                     PhaseSpaceVariable) 
                                     : _star_Bopp_monomial_A_times_B(to_Bopp, other, sgn)}
                                   )
    
    return _deprime(primed_out).expand()

class Star(sp.Expr, UnBoppable, Defined):
    """
    The s-parameterized star-product `A(q,p) ★ B(q,p) ★ ...` (or the `alpha` equivalent), 
    calculated using the Bopp shift.

    Parameters
    ----------

    *args
        The factors of the star-product, ordered from first to last. Since the algorithm
        utilizes the Bopp shift, only one operand be a non-polynomial.

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
        out += r"= f\left(\bm{\alpha}+\frac{s+1}{2}\partial_{\bm{\alpha}'},"
        out += r"\overline{\bm{\alpha}} + \frac{s-1}{2}\partial_{\overline{\bm{\alpha}}'}\right)"
        out += r"g\left(\bm{\alpha}',\overline{\bm{\alpha}}'\right)"
        return sp.Symbol(out)
    definition = _definition()
    
    def __new__(cls, *args : sp.Expr) -> sp.Expr:
        if not(args):
            return sp.Integer(1)
        
        unevaluated_args = []
        
        out = args[0]
        for k, arg in enumerate(args[1:]):
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
    
# Dual star product
###################

class _CannotDBoppFlag(TypeError):
    pass

def _dual_star_dBopp_monomial_A_times_B(A : sp.Expr, B : sp.Expr) -> sp.Expr:
    
    s = CahillGlauberS.val
    
    if A.is_Mul:
        args = A.args
    else:
        args = [A]
        
    non_psvo = sp.Integer(1)
    out = B
    
    # NOTE: Since operator products are not commutative, we need
    # to work from the rightmost argument or A.
    
    # NOTE: The dual star product is commutative so we only have one
    # variant of dBopp, i.e. we can always dBopp toward the right direction.
    
    for arg in reversed(args):
        if arg.has(createOp):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
                
            sub = list(arg.atoms(createOp))[0].sub
            ad = createOp(sub)
            
            for _ in range(n):
                out = ad*out + sp.Rational(1,2)*(s-1) * Commutator(ad, out)
                
        elif arg.has(annihilateOp):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
                
            sub = list(arg.atoms(annihilateOp))[0].sub
            a = annihilateOp(sub)
            
            for _ in range(n):
                out = a*out - sp.Rational(1,2)*(s+1) * Commutator(a, out)

        else:
            non_psvo *= arg
            
    return non_psvo*out

def _dual_star_base(F : sp.Expr, G : sp.Expr) -> sp.Expr:
    
    if not(F.has(PhaseSpaceVariableOperator) and G.has(PhaseSpaceVariableOperator)):
        return F*G
    
    if F.is_polynomial(annihilateOp, createOp):
        to_dBopp = F
        other = G
    elif G.is_polynomial(annihilateOp, createOp):
        to_dBopp = G
        other = F
    else:
        raise _CannotDBoppFlag
    
    def treat_add(A : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, functools.partial(_dual_star_dBopp_monomial_A_times_B, 
                                                           B=other)))
    
    return operation_routine(to_dBopp,
                             dStar,
                             [],
                             [PhaseSpaceObject],
                             {},
                             {sp.Add : treat_add,
                             (sp.Mul,
                              sp.Pow,
                              PhaseSpaceVariableOperator) 
                              : _dual_star_dBopp_monomial_A_times_B(to_dBopp, other)}
                              )

class dStar(sp.Expr, UnDBoppable, HilbertSpaceObject, Defined):
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
        
        # NOTE: dStar of two polynomials equals the s-ordering of their product,
        # since the iCGTransform of a product of two polynomials is that product,
        # converted into operators, and s-ordered. This helps shortcut the evaluation.
        
        undboppable_args = []
        
        out = sp.Integer(1)
        for k,arg in enumerate(args):
            try:
                if arg.has(qpType):
                    arg = qp2alpha(arg)
                out = _dual_star_base(out, arg)
            except _CannotBoppFlag:
                if out != 1:
                    undboppable_args.append(out)
                out = arg
                if k == (len(args)-1):
                    undboppable_args.append(arg)
                    
        if undboppable_args:
            return super().__new__(cls, *undboppable_args)
    
        return out
    
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\widetilde{\star}_s \left({%s}\right)" % sp.latex(arg)
        return out