import sympy as sp
import random

from typing import Sequence

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
def get_random_poly(objects : Sequence, 
                    coeffs : Sequence = [1], 
                    min_pow : int = 0, 
                    max_pow : int = 3, 
                    n_terms : int = 3) -> sp.Expr:
    """
    Make a random polynomial in ``objects``.
    
    Parameters
    ----------
    
    objects : sequence
        A sequence of the polynomial variables.
        
    coeffs : sequence, default : [1]
        A sequence of coefficients to choose from alongside each choice
        of object from ``objects``. As such, each output term has a coefficient that
        is a product of the elements of ``coeffs``.
        
    min_pow : int, default: 0
        Minimum number of ``objects`` a term may have. For example, setting ``min_pow=0`` 
        allows for constant terms to appear. Must not be negative.
        
    max_pow : int, default: 3
        Maximum number of ``objects`` a term may have, i.e., the polynomial degree. Must not
        be negative.
        
    n_terms : int, default: 3
        Number of terms to generate. Must not be negative.
    """
    if any(not(isinstance(x, (int,sp.Integer))) or x<0 for x in [min_pow, max_pow, n_terms]):
        raise ValueError(f"'min_pow', 'max_pow', and 'n_terms' must be nonnegative integers. Got {[min_pow, max_pow, n_terms]}")
    
    return sp.Add(*[sp.Mul(*[random.choice(coeffs)*random.choice(objects)
                             for _ in range(random.randint(min_pow, max_pow))])
                    for _ in range(n_terms)])
    
    
@preprocess_func
def collect_by_derivative(A : sp.Expr | sp.Equality) -> sp.Expr:
    """
    Collect terms containing the same ``sympy.Derivative`` object.

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