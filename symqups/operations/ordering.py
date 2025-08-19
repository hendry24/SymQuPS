import sympy as sp

from .. import s
from ..objects.base import qpTypePSO, alphaTypePSO, PhaseSpaceObject
from ..objects.scalars import alpha, alphaD, Scalar
from ..objects.cache import _sub_cache
from ..objects.operators import Operator, annihilateOp, createOp
from ..utils._internal import _operation_routine, _collect_alpha_type_oper_from_monomial
from ..utils.multiprocessing import _mp_helper
from ..utils.algebra import qp2a

def _normal_order_term_if_possible(expr : sp.Expr): 
    if not(expr.is_polynomial(Operator)):
        return expr
    
    def treat(A : sp.Expr):
        non_operator, collect_ad, collect_a = _collect_alpha_type_oper_from_monomial(A)
        return sp.Mul(non_operator, *collect_ad.values(), *collect_a.values())
    
    expr = qp2a(sp.sympify(expr))
    return _operation_routine(expr,
                              "_normal_ordering",
                              (Scalar, sp.Add),
                              (Operator,),
                              expr,
                              ((sp.Pow, Operator),
                                lambda A: A), 
                               ((sp.Mul,),
                                treat)
                              )

class sOrdering(sp.Expr):
    
    def __new__(cls, expr : sp.Expr, s : sp.Number = s.val):
        def make(A : sp.Expr):
            A = _normal_order_term_if_possible(A)
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
            non_operator, collect_ad, collect_a = _collect_alpha_type_oper_from_monomial(A)
            out = non_operator
            for sub in _sub_cache:
                ad, m = collect_ad[sub].args
                a, n = collect_a[sub].args
                if (m==0) or (n==0):
                    out *= ad**m * a*n
                else:
                    out *= make(ad*m * a*n)
            return out
     
        expr = qp2a(sp.sympify(expr))      
        return _operation_routine(expr,
                                  "sOrder",
                                  (Scalar,),
                                  (Operator,),
                                  expr,
                                  ((Operator,), lambda A: A),
                                  ((sp.Pow), treat_pow),
                                  ((sp.Function,), treat_foo),
                                  ((sp.Mul,), treat_mul),
                                  ((sp.Add,), lambda A: _mp_helper(A.args, sOrdering))
        )
        
    def _latex(self, printer):
        return r"\left\{ %s \right\}_{s=%s}" % (sp.latex(self.args[0]),
                                              sp.latex(self.args[1]))
        
    def expand(self, t = 1, define_special = False, **hints):
        """
        Expand the expression in terms of t-ordered expressions.
        By default, `t=1` corresponds to normal-ordering.
        """
        expr = super().expand(**hints)
        non_operator, collect_ad, collect_a = _collect_alpha_type_oper_from_monomial(expr)
        # should be 1
        
        if define_special:
            match self.args[1]:
                case -1:
                    return
        
        def expand_s_ordered_unipartite_string(sub):
            ad, m = collect_ad[sub].args
            a, n = collect_a[sub].args
            out = 0
            for k in range(min(m,n) + 1):
                out += (sp.factorial(k) * sp.binomial(m,k) * sp.binomial(n,k)
                        * ((t-s)/2)**k 
                        * sOrdering(ad**(m-k) * a**(n-k),
                                    s = t)) 
            
        
        
def s_ordered(sub, m, n):
    """
    Compute the s-ordering of the quantization of
    `alphaD(sub)**m * alpha(sub)**n`. Since we 
    can do whatever ordering when we quantize the
    expression, we choose normal ordering as the
    initial operator ordering, i.e. we initially
    have `createOp(sub)**m * annihilateOp(sub)**n`.
    Eq. (5.12) of https://doi.org/10.1103/PhysRev.177.1857
    can then be used to obtain its s-ordering.
    """
    
    if (m==0) and (n==0):
        return 1
    
    out = 0
    for k in range(min(m, n) + 1):
        out += (sp.factorial(k)
                * sp.binomial(n, k)
                * sp.binomial(m, k)
                * ((1-s.val)/2)**k
                * createOp(sub)**(m-k) * annihilateOp(sub)**(n-k)
        )
    return out