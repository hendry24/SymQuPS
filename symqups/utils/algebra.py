import sympy as sp
from sympy.core.function import UndefinedFunction

from ..objects.base import Base
from ..objects import scalars
from .multiprocessing import _mp_helper
from ._internal_routines import _operation_routine

__all__ = ["collect_by_derivative"]

def derivative_not_in_num(A : sp.Expr):
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
    
def define(expr):
    """
    Given a composite expression `expr`, call the `.define` method
    where applicable.
    """
    expr : sp.Expr = sp.sympify(expr)

    return expr.subs({A: A.define() for A in expr.atoms(Base)})
    