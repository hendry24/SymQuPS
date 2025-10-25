import sympy as sp
from typing import Sequence, Tuple

from .basic_routines import screen_type, is_nonconstant_polynomial
from .cache import sub_cache

from ..objects.operators import Operator, createOp, annihilateOp

# def decouple(expr : sp.Expr):
#     """
#     Decouple expressions when the noncommuting symbols actually belong
#     to different 'sub's.
#     """
#     # NOTE: WIP
        
#     if not(expr.has(Operator)):
#         return expr
    
#     # Power of Mul
#     def query(A : sp.Expr):
#         return (isinstance(A, sp.Pow) 
#                 and isinstance(A.args[0], sp.Mul) 
#                 and A.args[0].has(Operator))
#     def value(A : sp.Pow):
#         base = A.args[0]
#         base_separated_by_subs = separate_term_oper_by_sub()
#         exponent = A.args[1]
#         separate_term_oper_by_sub(base)
#         return sp.Mul(*[factor**exponent for factor in base_separated_by_subs])
#     expr = expr.replace(query, value)
    
#     # Functions whose 
#     functions = []
#     # expr = expr.replace(lambda A: isinstance(A, sp.exp) and isinstance())
    
#     return expr

def get_oper_sub(expr : sp.Expr) -> set[sp.Symbol]:
    return {atom.sub for atom in expr.atoms(Operator) if atom.has_sub}
            # Must use a set to avoid repeated 'sub's.

def is_universal(expr : sp.Expr) -> bool:
    """
    Returns 'True' if the input has at least one universally-noncommuting
    operator, e.g. 'densityOp'. Returns 'False' otherwise.
    """
    if not(expr.has(Operator)):
        return False
    return not(all(atom.has_sub for atom in expr.atoms(Operator)))

def separate_operator(expr: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
    expr = sp.sympify(expr)
    
    screen_type(expr, sp.Add, "_separate_operator")
    
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

def collect_alpha_type_oper_from_monomial_by_sub(expr : sp.Expr) -> Tuple[sp.Expr, dict, dict]:
    expr = sp.sympify(expr)
    
    screen_type(expr, sp.Add, "_collect_alpha_type_oper_from_monomial_by_sub")
    
    if not(expr.is_polynomial(annihilateOp, createOp)):
        raise ValueError("This function does not accept non-polynomials.")
    
    if isinstance(expr, sp.Mul):
        args = expr.args
    else:
        args = [expr]
    
    non_operator = sp.Number(1)
    collect_ad = {sub : [createOp(sub), sp.Number(0)] for sub in sub_cache}
    collect_a = {sub : [annihilateOp(sub), sp.Number(0)] for sub in sub_cache}
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

def separate_term_oper_by_sub(expr : sp.Expr) -> list[sp.Expr]:
    """
    Separate one term into a list of subexpressions, each 
    corresponding to one "sub group". 
    
    A sub group consists of one sub if the expression does 
    not have coupled quantities, and more than one sub otherwise.
    A coupled quantity looks something like `exp(a_1*a_2)`. Expressions
    such as `exp(a_1+a_2)` is also considered 
    
    Returns a list of 'expr.args' separated by 'sub' while maintaining 
    the order. That is, `Mul(*out)` returns the original expression.
    Scalars are multiplied into the first entry, which **always** contains
    non-operator objects.
    
    NOTE: This assumes that no universally-noncommuting operators (e.g., rho) 
    are present, which is usually the case in applications.
    """
    screen_type(expr, sp.Add, "_separate_term_oper_by_sub")
    
    if not(expr.has(Operator)):
        return [expr]
    
    if is_universal(expr):
        raise ValueError("Input must not contain universally noncommuting operators.")
     
    non_op, oper = separate_operator(expr)
    
    if not(isinstance(oper, sp.Mul)):
        return [non_op, oper]
    
    # The monkey-patched Mul is ordered by 'sub' groups. As such, an encounter
    # with an operator belonging to a different 'sub' group means to move
    # to the next entry in the output.

    out = [non_op, oper.args[0]]
    sub_group = get_oper_sub(oper.args[0])
    for factor in oper.args[1:]:
        subs_in_factor = list(get_oper_sub(factor))
                
        other_subs = []
        in_the_sub_group = False
        for factor_sub in subs_in_factor:
            if factor_sub in sub_group:
                in_the_sub_group = True
            else:
                other_subs.append(factor_sub)

        if in_the_sub_group:
            out[-1] *= factor
            sub_group.update(other_subs)
        else:
            # current factor belongs in the next 'sub_group'
            # and 'other_subs' contains all of the factor's 'sub's.
            sub_group = set(other_subs)
            out.append(factor)
    
    ####################
    # NOTE: Deprecated implementation. New one above is simpler given the monkey patch
    # to Mul. See '.mul.patched_Mul_flattern'.
    
    # out = [non_op]
    # sub_idx = {sub : None for sub in sub_cache}
    # for factor in oper.args:
        
    #     subs_in_factor = list(sp.ordered(get_oper_sub(factor)))
    #                                 # Turn into a sequence since it is used multiple times.
            
    #     if all(sub_idx[sub] is None for sub in subs_in_factor):
    #         """
    #         The given argument can go into its own slot in 'out'
    #         since there is no sub shared with other items
    #         already in the output (those with specified value in 'sub_idx').
    #         """
    #         for sub in subs_in_factor:
    #             sub_idx[sub] = len(out)
    #         out.append(factor)
        
    #     else:
    #         """
    #         If there is a specified sub in 'factor', then the factor
    #         shares at least one sub with an already existing item
    #         in 'out'. As such, we find the first shared sub and
    #         multiply 'factor' into that slot. While we are at it,
    #         we can collect 'sub's whose value in 'sub_idx' is still
    #         'None' to assign their values to be the same as the
    #         specified 'sub', telling the algorithm that future
    #         factors with these 'sub's already have a shared slot in
    #         'out'. 
            
    #         Additionally, since there may be more than one
    #         specified 'sub' in 'factor', we can use a flag such that
    #         further enconters with specified 'sub's are ignored by
    #         the algorithm.
    #         """
    #         sub_already_with_slot_in_out = None
    #         subs_with_unspecified_idx = []
    #         for sub in subs_in_factor:
    #             if sub_idx[sub] is None:
    #                 subs_with_unspecified_idx.append(sub)
    #             elif sub_already_with_slot_in_out is None:
    #                 sub_already_with_slot_in_out = sub
    #                 out[sub_idx[sub_already_with_slot_in_out]] *= factor        
                    
    #         for ssub in subs_with_unspecified_idx:
    #             sub_idx[ssub] = sub_idx[sub_already_with_slot_in_out]
    
    ###################
                
    return out
