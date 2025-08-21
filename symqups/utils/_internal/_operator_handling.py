import sympy as sp
from ._basic_routines import _screen_type
from..algebra import qp2a
from ...objects.operators import Operator, createOp, annihilateOp
from ...objects.cache import _sub_cache

def _separate_operator(expr: sp.Expr):
    expr = sp.sympify(expr)
    
    _screen_type(expr, sp.Add, "_separate_operator")
    
    if isinstance(expr, sp.Mul):
        args = expr.args
    else:
        args = [expr]
    
    non_operator = sp.Number(1)
    operator = sp.Number(1)
    for arg in args:
        if arg.has(Operator):
            operator *= arg
        else:
            non_operator *= arg
    
    # Even noncommuting symbols should commute with Operator,
    # so they go into non_operator.
    
    return non_operator, operator

def _separate_by_oper_polynomiality(expr : sp.Expr, polynomials_in = (createOp, annihilateOp)):
    """
    Subsequent elements of the output has alternating polynomiality in the 'polynomials_in'.
    """
    expr = sp.sympify(expr)
    
    _screen_type(expr, sp.Add, "_extract_alpha_type_operator_monomial_breaker")
    
    if not(isinstance(expr, sp.Mul)):
        return [expr]
    
    out = []
    factor = sp.Number(1)
    for arg in expr.args:
        if arg.is_polynomial(*polynomials_in) == factor.is_polynomial(*polynomials_in):
            factor *= arg
        else:
            if factor != 1:
                out.append(factor)
            factor = arg
    """
    The polynomiality may change at the last argument. If so, then 'factor'
    contains the last argument when the loop ends which has not been added
    to 'out' yet. Otherwise, 'factor' has not been appended to 'out' since
    the last ddetected polynomial change. So in any case, at the end of
    the loop, 'factor' contains the leftover arguments not appended yet to 'out',
    but it would never be unity. 
    """
    out.append(factor)
        
    return out

def _collect_alpha_type_oper_from_monomial(expr : sp.Expr):
    expr = qp2a(sp.sympify(expr))
    _screen_type(expr, sp.Add, "_collect_alpha_type_oper_from_monomial")
    
    if not(expr.is_polynomial(annihilateOp, createOp)):
        raise ValueError("This function does not accept non-polynomials.")
    
    if isinstance(expr, sp.Mul):
        args = expr.args
    else:
        args = [expr]
    
    non_operator = sp.Number(1)
    collect_ad = {sub : [createOp(sub), sp.Number(0)] for sub in _sub_cache}
    collect_a = {sub : [annihilateOp(sub), sp.Number(0)] for sub in _sub_cache}
    for A_ in args:
        if isinstance(A_, createOp):
            collect_ad[A_.sub][1] += 1
        elif A_.has(createOp): # Pow
            collect_ad[A_.args[0].sub][1] += A_.args[1]
        elif isinstance(A_, annihilateOp):
            collect_a[A_.sub][1] += 1
        elif A_.has(annihilateOp):
            collect_a[A_.args[0].sub][1] += A_.args[1]
        else:
            non_operator *= A_
            
    return non_operator, collect_ad, collect_a

def _collect_non_polynomial_by_sub(expr : sp.Expr):
    """
    'expr' is a Mul whose 'args' all 'has' 'Operator'.
    """
    if not(isinstance(expr, sp.Mul)):
        return [expr]
    
    non_op, oper = _separate_operator(expr)
    
    out = [non_op]
    sub_idx = {sub : None for sub in _sub_cache}
    for factor in oper.args:
        
        subs_in_factor = list(sp.ordered({oper.sub for oper in factor.atoms(Operator)}))
                                    # NOTE: must use a set to avoid repeated subs
                                    # and turn into a sequence since it is used multiple times.
        
        if all([sub is None for sub in subs_in_factor]):
            """
            The given argument can go into its own slot in 'out'
            since there is no sub shared with other items
            already in the output (those with specified value in 'sub_idx').
            """
            for sub in subs_in_factor:
                sub_idx[sub] = len(out)
            out.append(factor)
        
        else:
            """
            If there is a specified sub in 'factor', then the factor
            shares at least one sub with an already existing item
            in 'out'. As such, we find the first shared sub and
            multiply 'factor' into that slot. While we are at it,
            we can collect 'sub's whose value in 'sub_idx' is still
            'None' to assign their values to be the same as the
            specified 'sub', telling the algorithm that future
            factors with these 'sub's already have a shared slot in
            'out'. 
            
            Additionally, since there may be more than one
            specified 'sub' in 'factor', we can use a flag such that
            further enconters with specified 'sub's are ignored by
            the algorithm.
            """
            sub_already_with_slot_in_out = None
            subs_with_unspecified_idx = []
            for sub in subs_in_factor:
                if sub_idx[sub] is None:
                    subs_with_unspecified_idx.append(sub)
                elif sub_already_with_slot_in_out is None:
                    sub_already_with_slot_in_out = sub
                    out[sub_idx[sub_already_with_slot_in_out]] *= factor        
                    
            for ssub in subs_with_unspecified_idx:
                sub_idx[ssub] = sub_idx[sub_already_with_slot_in_out]
                
    return out