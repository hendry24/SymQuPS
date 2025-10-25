import sympy as sp
from typing import Callable, Dict, Union, Sequence, Any

from .multiprocessing import mp_helper

def treat_sub(sub, has_sub) -> sp.Symbol:
    """
    Treat the input subscript to always be a 'sympy.Symbol'.
    """
    if ((sub is None) or not(has_sub)):
        return sp.Symbol(r"")
    if isinstance(sub, str):
        return sp.Symbol(sub)
    if isinstance(sub, sp.Symbol):
        return sub
    return sp.Symbol(sp.latex(sub))

def screen_type(expr : sp.Expr, 
                forbidden_types : type | Sequence[type], 
                name : str) -> None:
    """
    Raise an error if the input 'expr' to some callable 'name' 
    **is** an object of 'forbidden_type'.
    """
    if not(isinstance(forbidden_types, Sequence)):
        forbidden_types = [forbidden_types]
    forbidden_types = tuple(forbidden_types)
    if isinstance(expr, forbidden_types):
        msg = f"'{name}' does not accept {forbidden_types}."
        raise TypeError(msg)
    
def only_allow_leaves_in_branches(expr : sp.Expr,
                              name : str,
                              leaves : tuple[type[sp.Basic]], 
                              allowed_branches : tuple[type[sp.Basic]]):
    """
    Ensure that all leaves occur only inside subexpressions of type `allowed_branches`.
    """
    if not(isinstance(leaves, Sequence)):
        leaves = (leaves,)
    leaves = tuple(leaves)
    
    if not(isinstance(allowed_branches, Sequence)):
        allowed_branches = (allowed_branches,)
    allowed_branches = tuple(allowed_branches)
    
    if any(not(leaf.is_Atom) for leaf in leaves):
        raise TypeError("'leaves' must all be atomic.")
        
    def walk_and_screen(A, branches, leaf):
        if isinstance(A, leaf):
            if not(any(isinstance(branch, allowed_branches)
                    for branch in branches)):
                msg = f"'{name}' only accepts '{leaf.__name__}' instances if they "
                msg += f"are contained within {str([cls.__name__ for cls in allowed_branches])}"
                raise ValueError(msg)
        for arg in A.args:
            walk_and_screen(arg, branches + (A,), leaf)
    
    for l in leaves:
        walk_and_screen(expr, (), l)
        # In the first recursion layer, the function checks if `expr`` is the screened 'leaf'
        # and raises an error if so. This is correct since we want 'leaf' to be contained in
        # 'allowed_branches'.
    
def deep_screen_type(expr : sp.Expr, 
                     forbidden_types : type | Sequence[type], 
                     name : str) -> None:
    """
    Raise an error if the input 'expr' to some callable 'name' 
    **contains** an object of 'forbidden_type'.
    """
    if not(isinstance(forbidden_types, tuple)):
        forbidden_types = [forbidden_types]
        
    if expr.has(*forbidden_types):
        msg = f"'{name} does not accept expressions that contain {forbidden_types}."
        raise TypeError(msg)

def invalid_input(inpt : object, name : str) -> None:
    """
    Raise an error for invalid 'inpt' to some callable 'name'.
    """
    msg = f"Invalid input to '{name}':\n"
    msg += r"%s" % sp.latex(inpt)
    raise ValueError(msg)

def operation_routine(expr : sp.Expr,
                       name : Union[str, Callable],
                       screen_types : Sequence[type],
                       deep_screen_types : Sequence[type],
                       return_if_expr_does_not_have : Dict[Union[type, Sequence[type]], 
                                                           Union[Callable, object]],
                       return_if_expr_is : Dict[Union[type, Sequence[type]], 
                                                Union[Callable, object]]) -> Any:
    """
    Routine that is used repeatedly in some functionalities, where the input expression
    is treated differently depending on its top-level algebraic strcuture, or where certain
    types of expressions/subexpressions are prohibited. 
    """
    
    if callable(name):
        name = name.__module__ + "." + name.__qualname__
    
    screen_type(expr, screen_types, name)
    deep_screen_type(expr, deep_screen_types, name)
    
    for if_does_not_have, then_return in return_if_expr_does_not_have.items():
        if not(isinstance(if_does_not_have, Sequence)):
            if_does_not_have = [if_does_not_have]      
        if not(expr.has(*if_does_not_have)):
            if callable(then_return):
                return then_return(expr)
            return then_return
    
    for if_is, then_return in return_if_expr_is.items():
        if isinstance(expr, if_is):
            if callable(then_return):
                return then_return(expr)
            return then_return
        
    invalid_input(expr, name)
    
def default_treat_add(summands : tuple[sp.Expr], foo : callable) -> sp.Expr:
    """
    Default 'sp.Add' treatment, where 'foo' is applied to each of
    'summands'.
    """
    return sp.Add(*mp_helper(summands, foo))

def is_nonconstant_polynomial(A, *gens):
    for gen in gens:
        if A.has(gen) and A.is_polynomial(gen):
            return True
    return False

def separate_term_by_polynomiality(expr : sp.Expr, 
                                   polynomials_in : tuple
                                   ) -> list[sp.Expr] :
    """
    Subsequent elements of the output has alternating polynomiality in the 'polynomials_in'.
    """
    
    if not(isinstance(polynomials_in, Sequence)):
        polynomials_in = [polynomials_in]
    
    screen_type(expr, sp.Add, separate_term_by_polynomiality)
    
    if not(isinstance(expr, sp.Mul)):
        return [expr]
    
    out = []
    factor = sp.Number(1)
    for arg in expr.args:
        if is_nonconstant_polynomial(arg, *polynomials_in) == is_nonconstant_polynomial(factor, *polynomials_in):
            factor *= arg
        else:
            if factor:
                out.append(factor)
            factor = arg
    
    # The polynomiality may change at the last argument. If so, then 'factor'
    # contains the last argument when the loop ends which has not been added
    # to 'out' yet. Otherwise, 'factor' has not been appended to 'out' since
    # the last ddetected polynomial change. So in any case, at the end of
    # the loop, 'factor' contains the leftover arguments not appended yet to 'out',
    # but it would never be unity. 
    
    out.append(factor)
        
    return out