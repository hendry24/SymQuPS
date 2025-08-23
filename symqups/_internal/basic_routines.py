import sympy as sp
from typing import Callable, Dict, Union, Sequence

def treat_sub(sub, has_sub):
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

def screen_type(expr : sp.Expr, forbidden_types : type | Sequence[type], name : str):
    """
    Raise an error if the input 'expr' to some callable 'name' 
    **is** an object of 'forbidden_type'.
    """
    forbidden_types = tuple(forbidden_types)
    if isinstance(expr, forbidden_types):
        msg = f"'{name}' does not accept {forbidden_types}."
        raise TypeError(msg)
    
def deep_screen_type(expr : sp.Expr, forbidden_types : type | Sequence[type], name : str):
    """
    Raise an error if the input 'expr' to some callable 'name' 
    **contains** an object of 'forbidden_type'.
    """
    if not(isinstance(forbidden_types, tuple)):
        forbidden_types = [forbidden_types]
        
    if expr.has(*forbidden_types):
        msg = f"'{name} does not accept expressions that contain {forbidden_types}."

def invalid_input(inpt : object, name : str):
    """
    Raise an error for invalid 'inpt' to some callable 'name'.
    """
    msg = f"Invalid input to '{name}':\n"
    msg += r"%s" % sp.latex(inpt)
    raise ValueError(msg)

def operation_routine(expr : sp.Expr,
                       name : str,
                       screen_types : Sequence[type],
                       deep_screen_types : Sequence[type],
                       return_if_expr_does_not_have : Dict[Union[type, Sequence[type]], 
                                                           Union[Callable, object]],
                       return_if_expr_is : Dict[Union[type, Sequence[type]], 
                                                Union[Callable, object]]):
    
    """
    Routine that is used repeatedly in some functionalities.
    """
    
    expr = sp.expand(sp.sympify(expr))
    
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