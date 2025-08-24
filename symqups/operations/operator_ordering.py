import sympy as sp
from itertools import permutations

from .. import s as ClahillGlauberS
from ..objects.scalars import Scalar
from .._internal.cache import sub_cache
from ..objects.operators import Operator, annihilateOp, createOp
from .._internal.basic_routines import operation_routine
from .._internal.operator_handling import (separate_operator,
                                           is_universal,
                                           get_oper_sub,
                                            collect_alpha_type_oper_from_monomial_by_sub,
                                            separate_term_by_polynomiality,
                                            separate_term_oper_by_sub)
from ..utils.multiprocessing import _mp_helper
from ..utils.algebra import qp2a

class sOrdering(sp.Expr):
    
    def __new__(cls, expr : sp.Expr, s : sp.Number | None = None, lazy : bool = False):
        expr = qp2a(sp.sympify(expr)) 
        
        # We assume that the input does not contain any universally-noncommuting
        # operators like 'densityOp'.
        if is_universal(expr):
            msg = "No universal operators should be put into s-ordering. "
            msg += "Input may contain 'densityOp' which never goes in "
            msg += "the ordering braces."
            raise ValueError(msg)
        
        s = sp.sympify(s)
        if s is None:
            s = ClahillGlauberS.val
        
        def make(A : sp.Expr):
            return super(sOrdering, cls).__new__(cls, A, s)
                    # need to specify since we do this
                    # inside another function. 

        def treat_add(A : sp.Expr):
            return sp.Add(*_mp_helper(A.args, sOrdering))
                    
        def treat_pow(A : sp.Expr):
            if A.is_polynomial(Operator):
                return A
            return make(A)
            
        def treat_foo(A : sp.Expr):
            if any(A.has(annihilateOp(sub)) and A.has(createOp(sub))
                   for sub in sub_cache):
                return make(A)
            return A
        
        def treat_mul(A : sp.Expr):
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
            # are normal-ordered to the "greatest extent possible". For example,
            # if there are non-polynomials stuck in the middle, then the
            # polynomials to the left and right are normal-ordered.
            
            bracket_arg_by_sub = separate_term_oper_by_sub(A)
                                # If 'A' is a monomial, then this only has one entry.
            
            out = bracket_arg_by_sub.pop(0) # gets all non-Operator subexpresions.
            
            for arg in bracket_arg_by_sub:
                
                if arg.is_polynomial(): 
                    
                    # contains only one sub because there are no coupled expressions.
                    
                    _, collect_ad, collect_a = \
                        collect_alpha_type_oper_from_monomial_by_sub(arg)
                    
                    arg_sub_lst = list(get_oper_sub(arg))
                    assert len(arg_sub_lst) == 1
                    
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
                                        
        return operation_routine(expr,
                                  "sOrder",
                                  [],
                                  [],
                                  {Operator : expr},
                                  {(Operator, sOrdering) : expr,
                                   sp.Pow : treat_pow,
                                   sp.Function : treat_foo,
                                   sp.Mul : treat_mul,
                                   sp.Add : treat_add}
                                  )
        
    def _latex(self, printer):
        return r"\left\{ %s \right\}_{s=%s}" % (sp.latex(self.args[0]),
                                              sp.latex(self.args[1]))

    def _collect_oper(self):
        non_operator, collect_ad, collect_a = \
            collect_alpha_type_oper_from_monomial_by_sub(self.args[0])
        assert non_operator == 1
        return collect_ad, collect_a
        
    def explicit(self):
        if not(self.args[0].is_polynomial(Operator)):
            return self
        
        collect_ad, collect_a = self._collect_oper()
        
        match self.args[1]:
            case -1:
                return sp.Mul(*[a**b for a,b in collect_a.values()], 
                              *[a**b for a,b in collect_ad.values()])
            case 0:
                out = 1
                for sub in _sub_cache:
                    ad, m = collect_ad[sub]
                    a, n = collect_a[sub]
                    to_permutate = [ad for _ in range(m)] + [a for _ in range(n)]
                    out_single_sub = 0
                    for permutation in permutations(to_permutate, len(to_permutate)):
                        out_single_sub += sp.Mul(*permutation)
                    if len(to_permutate) != 0:
                        out *= sp.cancel(out_single_sub / sp.factorial(len(to_permutate)))
                return out
            case 1:
                return _make_normal_ordered(collect_ad, collect_a)
            case default:
                return self

    def express(self, t = 1, explicit=True):
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
                    and self.args[1] in (-1, 0, 1)):
                    yy = yy.explicit()
                    
                out += (sp.factorial(k) * sp.binomial(m,k) * sp.binomial(n,k)
                        * ((t-self.args[1])/2)**k * yy)
            return out
        
        return sp.Mul(*[expand_s_ordered_unipartite_string(sub) for sub in sub_cache])

def normal_order(expr : sp.Expr):
    return sOrdering(expr, s=1).explicit()

def antinormal_order(expr : sp.Expr):
    return sOrdering(expr, s=-1).explicit()

def weyl_order(expr : sp.Expr):
    return sOrdering(expr, s=0).explicit()

def explicit(expr: sp.Expr):
    return expr.replace(lambda A: isinstance(A, sOrdering),
                        lambda A: A.explicit())
    
def express(expr : sp.Expr, t=1, explicit=True):
    return expr.replace(lambda A: isinstance(A, sOrdering),
                        lambda A: A.express(t=t, explicit=explicit))