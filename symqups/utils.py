import sympy as sp
from sympy.core.function import UndefinedFunction
import random

from ._internal.multiprocessing import mp_helper
from ._internal.cache import sub_cache

from .objects import scalars

###

def get_N():
    """
    Get the number of subsystems esablished so far in the session.
    """
    return len(sub_cache)

###

def get_random_poly(objects, coeffs=[1], max_pow=3, dice_throw=10) -> sp.Expr:
    """
    Make a random polynomial in 'objects'.
    """
    return sp.Add(*[sp.Mul(*[random.choice(coeffs)*random.choice(objects)**random.randint(0, max_pow)
                             for _ in range(dice_throw)])
                    for _ in range(dice_throw)])
    
###

def derivative_not_in_num(A : sp.Expr) -> sp.Expr:
    """
    Rewrite the expression such that the phase-space coordinates and derivatives with respect
    to them are not written on the numerator.
    """
    
    A = sp.sympify(A)
    
    if isinstance(A, sp.Add):
        return sp.Add(*mp_helper(A.args, derivative_not_in_num), evaluate=False)
    
    if not A.has(sp.Derivative):
        return A
    
    """
    `der_lst` here contains all the Derivative objects in a given term, 
    including the ones nested within another Derivative objects. We
    get the first one since it is the outermost.  
    """

    args_without_der = list(A.args)
    der_args = []
    for arg in A.args:
        if isinstance(arg, sp.Derivative):
            args_without_der.remove(arg)
            der_args.append(arg)
    non_der = sp.Mul(*args_without_der)
    der = sp.Mul(*der_args)
    
    with sp.evaluate(False):
        return sp.Mul(non_der, der)
    
def collect_by_derivative(A : sp.Expr, 
                          f : None | UndefinedFunction = None) -> sp.Expr:
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

    A = sp.expand(A)

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