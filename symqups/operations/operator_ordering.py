import sympy as sp
from itertools import permutations

from .. import s as ClahillGlauberS
from ..objects.scalars import Scalar
from ..objects.cache import _sub_cache
from ..objects.operators import Operator, annihilateOp, createOp
from ..utils._internal._basic_routines import _operation_routine
from ..utils._internal._operator_handling import (_separate_operator,
                                                  _separate_by_oper_polynomiality,
                                                  _collect_alpha_type_oper_from_monomial_by_sub)
from ..utils.multiprocessing import _mp_helper
from ..utils.algebra import qp2a

def _make_normal_ordered(col_ad : dict, 
                         col_a : dict):
    mul_ad = 1
    mul_a = 1
    for sub in _sub_cache:
        ad, m = col_ad[sub]
        mul_ad *= ad**m
        a, n = col_a[sub]
        mul_a *= a**n
    return sp.Mul(mul_ad, mul_a)

class sOrdering(sp.Expr):
    
    def __new__(cls, expr : sp.Expr, s : sp.Number | None = None, tidy : bool = False):
        expr = qp2a(sp.sympify(expr)) 
        
        s = sp.sympify(s)
        if s is None:
            s = ClahillGlauberS.val
        
        def make(A : sp.Expr):
            return super(sOrdering, cls).__new__(cls, A, s)
                    # need to specify since we do this
                    # inside another function. 
                    
        def treat_pow(A : sp.Expr):
            if A.args[1].has(createOp) and A.args[1].has(annihilateOp):
                return make(A)
            return A
            
        def treat_foo(A : sp.Expr):
            if A.has(createOp) and A.has(annihilateOp):
                return make(A)
            return A
        
        def treat_mul(A : sp.Expr):
            if not(tidy):
                leftovers, bracket_arg = _separate_operator(A)
                return leftovers * make(bracket_arg)
            
            if A.is_polynomial():
                non_operator, collect_ad, collect_a = _collect_alpha_type_oper_from_monomial_by_sub(A)
                out = non_operator
                for sub in _sub_cache:
                    ad, m = collect_ad[sub]
                    a, n = collect_a[sub]
                    if (m==0) or (n==0):
                        out *= ad**m * a**n
                    else:
                        out *= make(ad**m * a**n)
                return out
            else:
                leftovers, bracket_arg = _separate_operator(A)
                bracket_arg_by_polynomiality = _separate_by_oper_polynomiality(bracket_arg, 
                                                                               (createOp, annihilateOp))
                tidied_bracket_arg = 1
                for factor in bracket_arg_by_polynomiality:
                    if factor.is_polynomial():
                        non_operator, collect_ad, collect_a = \
                            _collect_alpha_type_oper_from_monomial_by_sub(factor)
                        assert non_operator == 1
                        tidied_bracket_arg *= _make_normal_ordered(collect_ad, collect_a)
                    else:
                        # If the nonpolynomial 'factor' does not share 'sub' with 
                        # the other factors, then we can savely separate its brackets
                        # or even write it out without brackets whence possible.
                        factor_sub = [atom.sub for atom in factor.atoms(Operator)]
                        bracket_arg_sub = [atom.sub for atom in bracket_arg.atoms(Operator)]
                        factor_is_separable = []
                        if True:
                            tidied_bracket_arg *= factor
                        else:
                            leftovers *= treat_foo(factor) # works as needed here.
                return leftovers * make(tidied_bracket_arg)
     
        def treat_add(A : sp.Expr):
            return sp.Add(*_mp_helper(A.args, sOrdering))
            
        return _operation_routine(expr,
                                  "sOrder",
                                  (Scalar,),
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
            _collect_alpha_type_oper_from_monomial_by_sub(self.args[0])
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

    def express(self, t = 1, explicit=True, **hints):
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
        
        return sp.Mul(*[expand_s_ordered_unipartite_string(sub) for sub in _sub_cache])
    
def normal_order(expr : sp.Expr):
    return sOrdering(expr, s=1).explicit()

def antinormal_order(expr : sp.Expr):
    return sOrdering(expr, s=-1).explicit()

def weyl_order(expr : sp.Expr):
    return sOrdering(expr, s=0).explicit()