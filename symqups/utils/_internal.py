import sympy as sp
from sympy.core.function import UndefinedFunction
from typing import Callable, Tuple

def _treat_sub(sub, has_sub):
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

def _screen_type(expr : sp.Expr, forbidden_type : object, name : str):
    """
    Raise an error if the input 'expr' to some callable 'name' 
    contains an object of 'forbidden_type'.
    """
    if expr.has(forbidden_type):
        msg = f"'{name}' does not accept '{forbidden_type}'"
        raise TypeError(msg)

def _invalid_input(inpt : object, name : str):
    """
    Raise an error for invalid 'inpt' to some callable 'name'.
    """
    msg = f"Invalid input to '{name}':\n"
    msg += r"%s" % sp.latex(inpt)
    raise ValueError(msg)

def _operation_routine(expr : sp.Expr,
                       name : str,
                       forbidden_types : tuple[type],
                       if_expr_does_not_have : tuple[type],
                       return_if_expr_does_not_have : Callable[[sp.Expr], sp.Expr],
                       *return_if_expr_is : tuple[Tuple[tuple[type], Callable[[sp.Expr], sp.Expr]]]):
    
    """
    Routine that is used repeatedly in some functionalities.
    """
    
    expr = sp.expand(sp.sympify(expr))
    
    _screen_type(expr, forbidden_types, name)
    
    if not(expr.has(*if_expr_does_not_have)):
        return return_if_expr_does_not_have(expr)
    
    for if_expr_is, then_return in return_if_expr_is:
        if isinstance(expr, if_expr_is):
            return then_return(expr)
        
    _invalid_input(expr, name)