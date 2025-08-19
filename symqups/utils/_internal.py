import sympy as sp
from typing import Callable, Tuple

from ..objects.cache import _sub_cache
from ..objects.operators import annihilateOp, createOp, Operator
from .algebra import qp2a

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

def _extract_alpha_type_operator_monomial_breaker(expr : sp.Expr):
    expr = qp2a(sp.sympify(expr))
    monomial_breaker = 1
    valid_monomial = 1
    
    _screen_type(expr, sp.Add, "_extract_alpha_type_operator_monomial_breaker")
    
    if not(isinstance(expr, sp.Mul)):
        if expr.is_polynomial(createOp, annihilateOp):
            valid_monomial *= expr
            return monomial_breaker, valid_monomial
        else:
            monomial_breaker *= expr
            return monomial_breaker, valid_monomial
    
    for arg in expr.args:
        if arg.is_polynomial():
            valid_monomial *= arg
        else:
            monomial_breaker *= arg
    
    return monomial_breaker, valid_monomial

def _collect_alpha_type_oper_from_monomial(expr : sp.Expr):
    expr = qp2a(sp.sympify(expr))
    assert expr.is_polynomial(annihilateOp, createOp)
    non_operator = 1
    collect_ad = {sub : 1 for sub in _sub_cache}
    collect_a = {sub : 1 for sub in _sub_cache}
    for A_ in expr.args:
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
            
    return non_operator, collect_ad, collect_a