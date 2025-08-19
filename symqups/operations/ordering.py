import sympy as sp

from .. import s
from ..objects.base import qpTypePSO, alphaTypePSO, PhaseSpaceObject
from ..objects.scalars import alpha, alphaD, Scalar
from ..objects.cache import _sub_cache
from ..objects.operators import Operator, annihilateOp, createOp
from ..utils._internal import _operation_routine, _invalid_input
from ..utils.multiprocessing import _mp_helper
from ..utils.algebra import qp2a

def _normal_order_term_if_possible(expr : sp.Expr):
    """
    expr is a single term
    """
    
    if expr.has(sp.Function):
        return expr
    
    def treat(A : sp.Expr):
        non_operator = 1
        collect_ad = {sub : 1 for sub in _sub_cache}
        collect_a = {sub : 1 for sub in _sub_cache}
        for A_ in A.args:
            if isinstance(A_, createOp):
                collect_ad[A_.sub] *= A_
            elif A_.has(createOp): # Pow
                collect_ad[A_.args[0].sub] *= A_
            elif isinstance(A_, annihilateOp):
                collect_a[A_.sub] *= A_
            elif A_.has(annihilateOp):
                collect_a[A_.args[0].sub] *= A_
            else:
                non_operator *= A_
        
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
    
    def __new__(cls, expr : sp.Expr):
        def make(A : sp.Expr):
            A = _normal_order_term_if_possible(A)
            return super(sOrdering, cls).__new__(cls, A)
                    # need to specify since we do this
                    # inside another function. 

        def treat_mul(A : sp.Expr):
            non_operator = 1
            operator = 1
            for A_ in A.args:
                if A_.has(Operator):
                    operator *= A_
                else:
                    non_operator *= A_
            return non_operator * make(operator)
     
        expr = qp2a(sp.sympify(expr))               
        return _operation_routine(expr,
                                  "sOrder",
                                  (Scalar,),
                                  (Operator,),
                                  expr,
                                  ((Operator, sp.Function, sp.Pow), 
                                    make),
                                  ((sp.Mul,),
                                    treat_mul),
                                  ((sp.Add,),
                                    lambda A: _mp_helper(A.args, sOrdering))
        )
        
    def _latex(self, printer):
        return r"\left\{ %s \right\}_{s=%s}" % (sp.latex(self.args[0]),
                                              sp.latex(s.val))
        
    def expand(self, t = 1, **hints):
        """
        Expand the expression in terms of t-ordered expressions.
        By default, `t=1` corresponds to normal-ordering.
        """
        expr = super().expand(**hints)
        
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