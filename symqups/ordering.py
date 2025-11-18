import sympy as sp
from itertools import permutations
from typing import Tuple
import functools

from ._internal.grouping import HilbertSpaceObject, CannotBoppShift, PhaseSpaceVariableOperator
from ._internal.cache import sub_cache
from ._internal.basic_routines import (operation_routine, 
                                       default_treat_add,)
from ._internal.math import (separate_operator,
                             has_universal_oper,
                             collect_alpha_type_oper_from_monomial_by_sub)
from ._internal.preprocessing import preprocess_class

from .objects.operators import Operator, annihilateOp, createOp

from .manipulations import qp2alpha, explicit_sOrdering

from . import s as CahillGlauberS

###

@preprocess_class
class sOrdering(sp.Expr, HilbertSpaceObject, CannotBoppShift):
    
    is_commutative = False
    
    def __new__(cls, expr : sp.Expr, s : sp.Number | None = None, lazy : bool = False) -> sp.Expr:
        expr = qp2alpha(sp.sympify(expr)) 
        
        # We assume that the input does not contain any universally-noncommuting
        # operators like 'densityOp'.
        if has_universal_oper(expr):
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
            return make(A, False)
            
        def treat_function(A : sp.Expr) -> sp.Expr:
            return make(A, False)
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
            if not(has_ordering_ambiguity(A)):
                return A
            
            if lazy:
                leftovers, bracket_arg = separate_operator(A)
                return leftovers * make(bracket_arg, 
                                        bracket_arg.is_polynomial(PhaseSpaceVariableOperator))
            
            # We don't care about operator ordering inside
            # the braces, so might as well return it pretty.
            coefs = []
            poly = {sub : [
                0, # number of polynomial ad_sub
                0, # number of polynomial a_sub
            ] for sub in sub_cache}
            nonpoly = [] 
            # We shove all nonpolynomial to the right of the polynomial part
            # since one nonpoly factor
            # may contain multiple subs. Separating "separable" factors
            # into their own ordering braces would be unnecessarily 
            # expensive, so we don't do that.
            #
            for arg in A.args:
                arg : sp.Expr
                if arg.has(PhaseSpaceVariableOperator):
                    if arg.is_polynomial(PhaseSpaceVariableOperator):
                        b, e = arg.as_base_exp()
                        if isinstance(b, createOp):
                            poly[b.sub][0] += e
                        elif isinstance(b, annihilateOp):
                            poly[b.sub][1] += e
                        else:
                            raise ValueError("Invalid value.")
                    else:
                        nonpoly.append(arg)
                else:
                    coefs.append(arg)
            
            in_braces = make(sp.Mul(*[createOp(sub)**pow_lst[0] for sub, pow_lst in poly.items()],
                                    *[annihilateOp(sub)**pow_lst[1] for sub, pow_lst in poly.items()],
                                    *nonpoly),
                             not(nonpoly))
            
            return sp.Mul(*coefs, in_braces)

        def make(A : sp.Expr, contains_poly : bool) -> sOrdering:
            if not(has_ordering_ambiguity(A)):
                return A
            obj = super(sOrdering, cls).__new__(cls, A, s)
            obj._contains_poly = contains_poly
            return obj
           
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
    
    @property
    def contains_poly(self):
        return self._contains_poly
        
    def _latex(self, printer) -> str:
        return r"\left\{ %s \right\}_{s=%s}" % (sp.latex(self.args[0]),
                                              sp.latex(self.args[1]))

    def _collect_oper(self) -> Tuple[dict, dict]:
        if not(self.contains_poly):
            raise RuntimeError("Cannot call '_collect_oper' when 'sOrdering' contains a nonpolynomial.")
        
        non_operator, collect_ad, collect_a = \
            collect_alpha_type_oper_from_monomial_by_sub(self.args[0])
        
        return collect_ad, collect_a
        
    def explicit(self) -> sp.Expr:
        if not(self.contains_poly):
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
        
        if not(self.contains_poly):
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
        
        return sp.Mul(*[expand_s_ordered_unipartite_string(sub) 
                        for sub in sub_cache])

def normal_order(expr : sp.Expr) -> sp.Expr:
    return explicit_sOrdering(sOrdering(expr, s=1))

def antinormal_order(expr : sp.Expr) -> sp.Expr:
    return explicit_sOrdering(sOrdering(expr, s=-1))

def weyl_order(expr : sp.Expr) -> sp.Expr:
    return explicit_sOrdering(sOrdering(expr, s=0))