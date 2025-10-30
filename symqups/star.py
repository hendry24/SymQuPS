import sympy as sp
import functools

from ._internal.grouping import (
    PhaseSpaceObject, HilbertSpaceObject,
    PhaseSpaceVariable, PhaseSpaceVariableOperator,
    CannotBoppShift, Defined, qpType
)
from ._internal.cache import sub_cache, primed_subs_dict, deprime_subs_dict
from ._internal.basic_routines import operation_routine
from ._internal.multiprocessing import mp_helper
from ._internal.preprocessing import preprocess_class

from .objects.base import Base
from .objects.scalars import alpha, alphaD, StateFunction
from .objects.operators import annihilateOp, createOp, densityOp

from .manipulations import qp2alpha

from . import s as CahillGlauberS

class _Primed(Base):
    def _get_symbol_name_and_assumptions(cls, psv):
        return (r"{%s}'" % sp.latex(psv),
                {"commutative" : False})
    
    def __new__(cls, 
                psv : PhaseSpaceVariable | PhaseSpaceVariableOperator):
    
        def make(A):
            return super(_Primed, cls).__new__(cls, A)
        
        def prime_expr(A : sp.Expr):
            return A.xreplace(primed_subs_dict)
        
        return operation_routine(psv,
                                 _Primed,
                                 [],
                                 [],
                                 {},
                                 {(PhaseSpaceVariable, 
                                   PhaseSpaceVariableOperator) : make,
                                  sp.Expr : prime_expr})
        
def _deprime(expr : sp.Expr):
    return expr.xreplace(deprime_subs_dict)

class _CannotBoppFlag(TypeError):
    pass

def _Bopp_monomial_A_times_B(A : sp.Expr,
                             B : sp.Expr,
                             left : bool,
                             hatted : bool):
    s = CahillGlauberS.val
    
    if A.is_Mul:
        args = A.args
    else:
        args = [A]
        
    non_psv = sp.Integer(1)
    out = _Primed(B)
    
    if hatted:
        a, ad = annihilateOp, createOp
        space_sgn = -1
    else:
        a, ad = alpha, alphaD
        space_sgn = 1
    
    if left:
        lr_sgn = -1
        iter_through = args
    else:
        lr_sgn = 1
        iter_through = reversed(args)
    
    for arg in iter_through:
        if arg.has(a):
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
                
            sub = list(arg.atoms(a))[0].sub
            
            for _ in range(n):
                if left:
                    new_out = out*a(sub)
                else:
                    new_out = a(sub)*out
                new_out += space_sgn*sp.Rational(1,2)*(s+lr_sgn)*sp.Derivative(out, _Primed(ad(sub)))
                
                out = new_out
        
        elif arg.has(ad):
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
                
            sub = list(arg.atoms(ad))[0].sub
            
            for _ in range(n):
                if left:
                    new_out = out*ad(sub)
                else:
                    new_out = ad(sub)*out
                new_out += space_sgn*sp.Rational(1,2)*(s-lr_sgn)*sp.Derivative(out, _Primed(a(sub)))
                
                out = new_out
                
        else:
            non_psv *= arg
            
    return _deprime((non_psv*out).doit().expand())

def _star_base(F : sp.Expr, 
               G : sp.Expr,
               hatted : bool):
    
    if hatted:
        var_group = HilbertSpaceObject
        a, ad = annihilateOp, createOp
        state = densityOp
    else:
        var_group = PhaseSpaceObject
        a, ad = alpha, alphaD
        state = StateFunction
    
    if not(F.has(var_group)) or not(G.has(var_group)):
        return F*G
    
    if F.has(CannotBoppShift) and G.has(CannotBoppShift):
        raise _CannotBoppFlag
    
    if F.is_polynomial(a, ad):
        Bopp_shifted = F
        other = G
        left = False
    elif G.is_polynomial(a, ad):
        Bopp_shifted = G
        other = F
        left = True
    else:
        raise _CannotBoppFlag
    
    def treat_add(A : sp.Add):
        return sp.Add(*mp_helper(A.args,
                                 functools.partial(_Bopp_monomial_A_times_B,
                                                   B=other,
                                                   left=left,
                                                   hatted=hatted)
                                 )
                      )
        
    return operation_routine(Bopp_shifted,
                             _StarTemplate,
                             [],
                             [],
                             {},
                             {sp.Add : treat_add,
                              (sp.Mul,
                               sp.Pow,
                               var_group,
                               state) 
                              : _Bopp_monomial_A_times_B(Bopp_shifted, 
                                                         other, 
                                                         left, 
                                                         hatted)
                              }
                             )

@preprocess_class
class _StarTemplate(sp.Expr, CannotBoppShift, Defined):
   
    @staticmethod
    def _definition():
        return NotImplemented
    definition = _definition()
    
    def __new__(cls, *args : sp.Expr, hatted : bool):
        unevaluated_args = []
        
        out = sp.Integer(1)
        for k, arg in enumerate(args):
            try:
                if arg.has(qpType):
                    arg = qp2alpha(arg)
                out = _star_base(out, arg, hatted)
            except _CannotBoppFlag:
                if out != 1:
                    unevaluated_args.append(out)
                out = arg
                if k == (len(args)-1):
                    unevaluated_args.append(arg)
                    
        if unevaluated_args:
            obj = super().__new__(cls, *unevaluated_args)
            obj._hatted = hatted
            return obj
        
        return out
    
    @property
    def hatted(self):
        return self._hatted
    
    def _latex(self, printer):
        out = ""
        oper = r"\mathbin{\widehat{\star}_s}" if self.hatted else r"\mathbin{\star_s}"
        for k, arg in enumerate(self.args):
            if k != 0:
                out += oper
            out += r"\left({%s}\right)" % sp.latex(arg)
        return out
    
class Star(_StarTemplate):
    def __new__(cls, *args : sp.Expr):
        return super().__new__(cls, *args, hatted=False)

class HattedStar(_StarTemplate):
    def __new__(cls, *args : sp.Expr):
        return super().__new__(cls, *args, hatted=True)