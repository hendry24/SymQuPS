import sympy as sp
import random
from sympy.core.function import UndefinedFunction

from ..objects.base import Base
from ..objects.cache import _sub_cache
from ..objects import scalars
from ..objects.operators import qOp, pOp, annihilateOp, createOp, Operator
from .multiprocessing import _mp_helper

def get_random_poly(objects, coeffs=[1], max_pow=3, dice_throw=10):
    """
    Make a random polynomial in 'objects'.
    """
    return sp.Add(*[sp.Mul(*[random.choice(coeffs)*random.choice(objects)**random.randint(0, max_pow)
                             for _ in range(dice_throw)])
                    for _ in range(dice_throw)])

def define(expr : sp.Expr) -> sp.Expr:
    """
    Given a composite expression `expr`, call the `.define` method
    where applicable.
    """
    expr = sp.sympify(expr)
    expr_defined = expr.subs({A: A.define() for A in expr.atoms(Base)})
    return sp.expand(expr_defined)

def qp2a(expr : sp.Expr) -> sp.Expr:
    def get_subs_expr(A : scalars.Scalar | Operator):
        if isinstance(A, scalars.Scalar):
            a, ad = scalars.alpha(A.sub), scalars.alphaD(A.sub)
        else:
            a, ad = annihilateOp(A.sub), createOp(A.sub)
            
        mu = scalars.mu
        mu_conj = sp.conjugate(mu)
        hbar = scalars.hbar
        
        if isinstance(A, (scalars.q, qOp)):
            out = mu*a + mu_conj*ad
        else:
            out = sp.I*mu*mu_conj*(mu*ad - mu_conj*a)
            
        out *= sp.sqrt(2*hbar) / (mu**2 + mu_conj**2)
        
        return out
        
    sub_dict = {}
    for sub in _sub_cache:
        sub_dict[scalars.q(sub)] = get_subs_expr(scalars.q(sub))
        sub_dict[scalars.p(sub)] = get_subs_expr(scalars.p(sub))
        sub_dict[qOp(sub)] = get_subs_expr(qOp(sub))
        sub_dict[pOp(sub)] = get_subs_expr(pOp(sub))
        
    return sp.expand(expr.subs(sub_dict))

def derivative_not_in_num(A : sp.Expr) -> sp.Expr:
    """
    Rewrite the expression such that the phase-space coordinates and derivatives with respect
    to them are not written on the numerator.
    """
    
    A = sp.sympify(A)
    
    if isinstance(A, sp.Add):
        return sp.Add(*_mp_helper(A.args, derivative_not_in_num), evaluate=False)
    
    der_lst = list(A.find(sp.Derivative))
    if not(der_lst):
        return A
    
    """
    `der_lst` here contains all the Derivative objects in a given term, 
    including the ones nested within another Derivative objects. We
    get the first one since it is the outermost.  
    """

    Q_args_without_der = list(A.args)
    Q_args_without_der.remove(der_lst[0])
    
    return sp.Mul(sp.Mul(*Q_args_without_der), der_lst[0], evaluate=False)
    
def collect_by_derivative(A : sp.Expr, 
                          f : None | UndefinedFunction = None) \
    -> sp.Expr:
    """
    Collect terms by the derivatives of the input function, by default those of the Wigner function `W`.

    Parameters
    ----------

    A : sympy object
        Quantity whose terms is to be collected. If `A` contains no
        function, then it is returned as is. 

    f : sympy.Function, default: `W`
        Function whose derivatives are considered.

    Returns
    -------

    out : sympy object
        The same quantity with its terms collected. 
    """

    A = A.expand()

    if not(A.atoms(sp.Function)):
        return A

    q = scalars.q()
    p = scalars.p()
    if f is None:
        f = scalars.W()

    max_order = max([A_.derivative_count 
                     for A_ in list(A.atoms(sp.Derivative))]+[0])

    def dq_m_dp_n(m, n):
        if m==0 and n==0:
            return f
        return sp.Derivative(f, 
                             *[q for _ in range(m)], 
                             *[p for _ in range(n)])
    
    return sp.collect(A, [dq_m_dp_n(m, n) 
                          for m in range(max_order) 
                          for n in range(max_order - m)])