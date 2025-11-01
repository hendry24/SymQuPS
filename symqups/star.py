import sympy as sp
import functools
import itertools

from ._internal.grouping import (
    PhaseSpaceObject, HilbertSpaceObject,
    CannotBoppShift, Defined, qpType
)
from ._internal.basic_routines import operation_routine
from ._internal.multiprocessing import mp_helper
from ._internal.preprocessing import preprocess_class

from .objects.scalars import alpha, alphaD, StateFunction
from .objects.operators import annihilateOp, createOp, densityOp

from .manipulations import qp2alpha, dagger

from . import s as CahillGlauberS

###

class _CannotBoppFlag(TypeError):
    pass

###

def _Star_Bopp_monomial_A_times_B(A : sp.Expr, B : sp.Expr, left : bool):
    s = CahillGlauberS.val
    
    if A.is_Mul:
        args = A.args
    else:
        args = [A]
    
    if left:
        lr_sgn = -1
    else:
        lr_sgn = 1
    
    coef_factors = []
    b_lst = []
    e_lst = []
    for arg in args:
        if arg.has(alpha, alphaD):
            b, e = arg.as_base_exp()
            b_lst.append(b)
            e_lst.append(e)
        else:
            coef_factors.append(arg)
    
    bopped_series_summands = []
    for j_lst in itertools.product(*[range(e) for e in e_lst]):
        bopp_factors = []
        der_wrt = []
        for j, b, e in zip(j_lst, b_lst, e_lst):
            if isinstance(b, alpha):
                xi = sp.Rational(1,2) * (s+lr_sgn)
            elif isinstance(b, alphaD):
                xi = sp.Rational(1,2) * (s-lr_sgn)
            else:
                raise TypeError("Invalid type. Contact dev.")
            bopp_factors.extend([sp.binomial(e, j),
                                 b**(e-j),
                                 xi**j])
            der_wrt.append((dagger(b), j))
        
        der = sp.Derivative(B, *der_wrt).doit()
        y = sp.Mul(*coef_factors, *bopp_factors, der)
        bopped_series_summands.append(y)
    
    return sp.Add(*bopped_series_summands)

###

def _HattedStar_Bopp_monomial_A_times_B(A : sp.Expr, B : sp.Expr, left : bool):
    s = CahillGlauberS.val
    
    if A.is_Mul:
        args = A.args
    else:
        args = [A]
    
    if left:
        lr_sgn = -1
        iter_through = args
    else:
        lr_sgn = 1
        iter_through = reversed(args)
        
    xi_a = -sp.Rational(1,2) * (s + lr_sgn)
    xi_ad = -sp.Rational(1,2) * (s - lr_sgn)
    # xi_a is the coefficient attached to the Bopp shift of a,
    # likewise for xi_ad
    
    coef_factors = []
    op = []
    op_bopp = []
    for arg in iter_through:
        if arg.has(annihilateOp, createOp):
            b, e = arg.as_base_exp()
            op.append(b)
            
            xi = xi_a if isinstance(b, annihilateOp) else xi_ad
            op_bopp.append([xi, (dagger(b), e)])
        
        else:
            coef_factors.append(arg)
    
    expansion_combos = itertools.product(*[(x,y) 
                                           for x,y in zip(op, op_bopp)])
    
    ###
    
    out_summands = []
    for combo in expansion_combos:
        xi = []
        op = []
        diff_B_wrt = []
        for o in combo:
            if isinstance(o, list):
                xi.append(o[0])
                diff_B_wrt.append(o[1])
            else:
                op.append(o)
        out_summands.append(sp.Mul(*coef_factors,
                                   *xi,
                                   *op,
                                   sp.Derivative(B, *diff_B_wrt)))
        
    return sp.Add(*out_summands)
    
###

def _star_base(F : sp.Expr, 
               G : sp.Expr,
               hatted : bool):
    
    if hatted:
        var_group = HilbertSpaceObject
        a, ad = annihilateOp, createOp
        state = densityOp
        bopp_monomial = _HattedStar_Bopp_monomial_A_times_B
    else:
        var_group = PhaseSpaceObject
        a, ad = alpha, alphaD
        state = StateFunction
        bopp_monomial = _Star_Bopp_monomial_A_times_B
    
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
                                 functools.partial(bopp_monomial,
                                                   B=other,
                                                   left=left)
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
                              : lambda A: bopp_monomial(A,
                                                        B=other,
                                                        left=left)
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