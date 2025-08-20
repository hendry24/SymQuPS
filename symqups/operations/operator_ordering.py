import sympy as sp
from itertools import permutations

from .. import s as ClahillGlauberS
from ..objects.scalars import Scalar
from ..objects.cache import _sub_cache
from ..objects.operators import Operator, annihilateOp, createOp
from ..utils._internal._basic_routines import _operation_routine
from ..utils._internal._operator_handling import (_separate_operator,
                                                  _separate_by_oper_polynomiality,
                                                  _collect_alpha_type_oper_from_monomial,
                                                  _normal_order_alpha_type_oper_monomial)
from ..utils.multiprocessing import _mp_helper
from ..utils.algebra import qp2a

class sOrdering(sp.Expr):
    
    def __new__(cls, expr : sp.Expr, s : sp.Number | None = None):
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
            if A.is_polynomial():
                non_operator, collect_ad, collect_a = _collect_alpha_type_oper_from_monomial(A)
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
                A_nonop, A_op = _separate_operator(A)
                if A_op == 1:
                    return A_nonop
                A_op_by_polynomiality = _separate_by_oper_polynomiality(A_op, (createOp, annihilateOp))
                reordered_A_op = 1
                for factor in A_op_by_polynomiality:
                    if factor.is_polynomial():
                        reordered_A_op *= _normal_order_alpha_type_oper_monomial(factor)
                    else:
                        reordered_A_op *= factor
                return A_nonop * make(reordered_A_op)
     
        return _operation_routine(expr,
                                  "sOrder",
                                  (Scalar,),
                                  (Operator,),
                                  lambda A: expr,
                                  ((Operator,), lambda A: A),
                                  ((sp.Pow), treat_pow),
                                  ((sp.Function,), treat_foo),
                                  ((sp.Mul,), treat_mul),
                                  ((sp.Add,), lambda A: _mp_helper(A.args, sOrdering))
        )
        
    def _latex(self, printer):
        return r"\left\{ %s \right\}_{s=%s}" % (sp.latex(self.args[0]),
                                              sp.latex(self.args[1]))

    def _collect_oper(self):
        non_operator, collect_ad, collect_a = _collect_alpha_type_oper_from_monomial(self.args[0])
        assert non_operator == 1
        return collect_ad, collect_a
        
    def define(self):
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
                return sp.Mul(*[a**b for a,b in collect_ad.values()], 
                              *[a**b for a,b in collect_a.values()])
            case default:
                return self

    def express(self, t = 1, define=True, **hints):
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
                if (define 
                    and isinstance(yy, sOrdering)
                    and self.args[1] in (-1, 0, 1)):
                    yy = yy.define()
                    
                out += (sp.factorial(k) * sp.binomial(m,k) * sp.binomial(n,k)
                        * ((t-self.args[1])/2)**k * yy)
            return out
        
        return sp.Mul(*[expand_s_ordered_unipartite_string(sub) for sub in _sub_cache])