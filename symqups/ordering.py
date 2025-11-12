import sympy as sp
from itertools import permutations
from typing import Tuple
import warnings
import functools

from ._internal.grouping import HilbertSpaceObject, CannotBoppShift
from ._internal.cache import sub_cache
from ._internal.basic_routines import (operation_routine, 
                                       default_treat_add, 
                                       separate_term_by_polynomiality)
from ._internal.math import (separate_operator,
                                           is_universal,
                                           get_oper_sub,
                                            collect_alpha_type_oper_from_monomial_by_sub,
                                            separate_term_oper_by_sub)
from ._internal.preprocessing import preprocess_class

from .objects.scalars import q, p, alpha, alphaD
from .objects.operators import Operator, annihilateOp, createOp

from .manipulations import qp2alpha

from . import s as CahillGlauberS

###

@preprocess_class
class sOrdering(sp.Expr, HilbertSpaceObject, CannotBoppShift):
    
    is_commutative = False
    
    def __new__(cls, expr : sp.Expr, s : sp.Number | None = None, lazy : bool = False) -> sp.Expr:
        expr = qp2alpha(sp.sympify(expr)) 
        
        # We assume that the input does not contain any universally-noncommuting
        # operators like 'densityOp'.
        if is_universal(expr):
            msg = "No universal operators should be put into s-ordering. "
            msg += "Input may contain 'densityOp' which never goes in "
            msg += "the ordering braces."
            raise ValueError(msg)
            
        s = sp.sympify(s)
        if s is None:
            s = CahillGlauberS.val
    
        def has_ordering_ambiguity(A : sp.Expr) -> bool:
            if any(A.has(annihilateOp(sub)) and A.has(createOp(sub))
                   for sub in sub_cache):
                return True
            return False
        
        def treat_add(A : sp.Expr) -> sp.Expr:
            return default_treat_add(A.args, functools.partial(sOrdering,
                                                               s=s,
                                                               lazy=lazy))
                    
        def treat_pow(A : sp.Expr) -> sp.Expr:
            if A.is_polynomial(Operator):
                return A
            return make(A)
            
        def treat_function(A : sp.Expr) -> sp.Expr:
            if has_ordering_ambiguity(A):
                return make(A)
            return A
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
            if lazy:
                leftovers, bracket_arg = separate_operator(A)
                return leftovers * make(bracket_arg)
            
            # We don't care about operator ordering inside
            # the braces, so might as well return it pretty.
            #
            # Here, factors belonging to different 'sub's go in their
            # own braces. If there are "coupled factors", such as exp(a_1*a_2),
            # then the factors corresponding to the "coupled 'sub's" are
            # enclosed by the same braces. The expression inside the braces
            # are normal-ordered to the "greatest extent possible":
            # the polynomial parts are collected and normal-ordered, and
            # the nonpolynomial parts go after the poynomial parts.
            
            bracket_arg_by_sub = separate_term_oper_by_sub(A)
                                            
            out = bracket_arg_by_sub.pop(0) # gets all non-Operator subexpresions.
            
            for arg in bracket_arg_by_sub:
                
                if not(has_ordering_ambiguity(arg)):
                    out *= arg
                  
                elif arg.is_polynomial(): 
                    
                    # contains only one sub because there are no coupled expressions.
                    
                    _, collect_ad, collect_a = \
                        collect_alpha_type_oper_from_monomial_by_sub(arg)
                    
                    arg_sub_lst = list(get_oper_sub(arg))
                    
                    arg_sub = arg_sub_lst[0]
                    
                    ad, m = collect_ad[arg_sub]
                    a, n = collect_a[arg_sub]
                    
                    if (m==0) or (n==0):
                        out *= ad**m * a**n
                    else:
                        out *= make(ad**m * a**n)
                    
                else:
                    
                    arg_by_polynomiality = separate_term_by_polynomiality(arg,
                                                                          (Operator,))
                    collect_polynomial = sp.Number(1)
                    collect_nonpolynomial = sp.Number(1)
                    for argg in arg_by_polynomiality:
                        if argg.is_polynomial():
                            collect_polynomial *= argg
                        else:
                            collect_nonpolynomial *= argg
                    
                    _, collect_ad, collect_a = \
                        collect_alpha_type_oper_from_monomial_by_sub(collect_polynomial)
                        
                    collect_polynomial_normal_ordered = sp.Number(1)
                    for sub in get_oper_sub(arg):
                        collect_polynomial_normal_ordered *= \
                            sp.Pow(*collect_ad[sub]) * sp.Pow(*collect_a[sub])
                        
                    out *= make(collect_polynomial_normal_ordered * collect_nonpolynomial)
                        
            return out

        def make(A : sp.Expr) -> sOrdering:
            return super(sOrdering, cls).__new__(cls, A, s)
           
        return operation_routine(sp.expand(sp.sympify(expr)),
                                  sOrdering,
                                  [],
                                  [],
                                  {Operator : expr},
                                  {(Operator, sOrdering) : expr,
                                   sp.Pow : treat_pow,
                                   sp.Function : treat_function,
                                   sp.Mul : treat_mul,
                                   sp.Add : treat_add}
                                  )
    
    @property
    def s_val(self):
        return self.args[1]
        
    def _latex(self, printer) -> str:
        return r"\left\{ %s \right\}_{s=%s}" % (sp.latex(self.args[0]),
                                              sp.latex(self.args[1]))

    def _collect_oper(self) -> Tuple[dict, dict]:
        non_operator, collect_ad, collect_a = \
            collect_alpha_type_oper_from_monomial_by_sub(self.args[0])
        assert non_operator == 1
        return collect_ad, collect_a
        
    def explicit(self) -> sp.Expr:
        if not(self.args[0].is_polynomial(Operator)):
            return self
        
        collect_ad, collect_a = self._collect_oper()
        
        match self.args[1]:
            case -1:
                return sp.Mul(*[a**b for a,b in collect_a.values()], 
                              *[a**b for a,b in collect_ad.values()])
            case 0:
                out = 1
                for sub in sub_cache:
                    ad, m = collect_ad[sub]
                    a, n = collect_a[sub]
                    to_permutate = [ad]*m + [a]*n
                    out_single_sub = 0
                    for permutation in permutations(to_permutate, len(to_permutate)):
                        out_single_sub += sp.Mul(*permutation)
                    if len(to_permutate) != 0:
                        out *= sp.cancel(out_single_sub / sp.factorial(len(to_permutate)))
                return out
            case 1:
                return sp.Mul(*[a**b for a,b in collect_ad.values()], 
                              *[a**b for a,b in collect_a.values()])
            case default:
                return self

    def express(self, t = 1, explicit=True) -> sp.Expr:
        """
        Expand the expression in terms of t-ordered expressions.
        By default, `t=1` corresponds to normal-ordering. If `define`,
        then the expanded t-ordered expressiosn are defined when possible. 
        By default, express the object in terms of normal-ordered products.
        """
        
        if not(self.args[0].is_polynomial(Operator)):
            return self
        
        collect_ad, collect_a = self._collect_oper()                            
        
        def expand_s_ordered_unipartite_string(sub):
            ad, m = collect_ad[sub]
            a, n = collect_a[sub]
            out = 0
            for k in range(min(m,n) + 1):
                yy = sOrdering(ad**(m-k) * a**(n-k), s=t)
                if (explicit 
                    and isinstance(yy, sOrdering)
                    and t in (-1, 0, 1)):
                    yy = yy.explicit()
                    
                out += (sp.factorial(k) * sp.binomial(m,k) * sp.binomial(n,k)
                        * ((t-self.args[1])/2)**k * yy)
            return out
        
        return sp.Mul(*[expand_s_ordered_unipartite_string(sub) for sub in sub_cache])

def normal_order(expr : sp.Expr) -> sp.Expr:
    return sOrdering(expr, s=1).explicit()

def antinormal_order(expr : sp.Expr) -> sp.Expr:
    return sOrdering(expr, s=-1).explicit()

def weyl_order(expr : sp.Expr) -> sp.Expr:
    return sOrdering(expr, s=0).explicit()