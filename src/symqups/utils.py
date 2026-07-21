import sympy as sp
import random

from ._internal.cache import sub_cache
from ._internal.preprocessing import preprocess_func

###

def get_N():
    """
    Get the number of subsystems esablished so far in the session.
    """
    return len(sub_cache)

###

@preprocess_func
def get_random_poly(objects, coeffs=[1], min_pow =0, max_pow=3, n_terms=3) -> sp.Expr:
    """
    Make a random polynomial in 'objects'.
    """
    return sp.Add(*[sp.Mul(*[random.choice(coeffs)*random.choice(objects)
                             for _ in range(random.randint(min_pow, max_pow))])
                    for _ in range(n_terms)])
    
###

@preprocess_func
def derivative_not_in_num(A : sp.Expr) -> sp.Expr:
    """
    Rewrite the expression such that the phase-space coordinates and derivatives with respect
    to them are not written on the numerator.
    """
    
    if isinstance(A, sp.Add):
        return sp.Add(*[derivative_not_in_num(arg) for arg in A.args], evaluate=False)
    
    if not A.has(sp.Derivative):
        return A
    
    # `der_lst` here contains all the Derivative objects in a given term, 
    # including the ones nested within another Derivative objects. We
    # get the first one since it is the outermost.

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
    
@preprocess_func
def collect_by_derivative(A : sp.Expr | sp.Equality) -> sp.Expr:
    """
    Collect terms by the derivatives of the input function, by default those of the state function.

    Parameters
    ----------

    A : sympy.Expr or sympy.Equality
        Expression whose terms is to be collected.

    Returns
    -------

    out : sympy.Expr or sympy.Equality
        The same quantity with its terms collected. 
    """

    if A.is_Equality:
        return sp.Equality(collect_by_derivative(A.lhs),
                           collect_by_derivative(A.rhs))

    return sp.collect(A,
                      list(A.atoms(sp.Derivative)))