import sympy as sp
from typing import Sequence, Tuple
from functools import cmp_to_key

from .basic_routines import screen_type
from .cache import sub_cache
from .grouping import NotAnOperator

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

def separate_term_by_polynomiality(expr : sp.Expr, 
                                   polynomials_in : tuple[Operator] = (createOp, annihilateOp)
                                   ) -> list[sp.Expr] :
    """
    Subsequent elements of the output has alternating polynomiality in the 'polynomials_in'.
    """
    expr = sp.sympify(expr)
    
    if not(isinstance(polynomials_in, Sequence)):
        polynomials_in = [polynomials_in]
    
    screen_type(expr, sp.Add, "_separate_term_oper_by_polynomiality")
    
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
    
    # The polynomiality may change at the last argument. If so, then 'factor'
    # contains the last argument when the loop ends which has not been added
    # to 'out' yet. Otherwise, 'factor' has not been appended to 'out' since
    # the last ddetected polynomial change. So in any case, at the end of
    # the loop, 'factor' contains the leftover arguments not appended yet to 'out',
    # but it would never be unity. 
    
    out.append(factor)
        
    return out

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

###

global original_Mul_flatten
original_Mul_flatten = sp.Mul.flatten

# HACK: Monkey patching sympy's core implementation to do something
# extra if 'Operator' objects are input.
def patched_Mul_flatten(seq : Sequence) -> Tuple[list, list, list]: 

    # This patch reorders the noncommuting 'Operator' objects according
    # to sympy's canonical ordering of 'sub_cache', resulting in
    # prettier outputs in multipartite cases.
    #
    # Other noncommuting object classes within the package are '_Primed'
    # and _DerivativeSymbol. These, however, do not need the reordering
    # treatment as they are used internally by the package. Since these
    # and 'Operator' belong to different formulations, they should never
    # be in the same expression. Here we reorder only 
    # expressions where Operator appears, since they are the one whose 
    # ordering we care about.
    
    ###
    
    # First, the package forgoes the reordering process in the cases where we 
    # are clearly not multiplying operators with operators. The first
    # conditional by itself is not sufficient since we may have 
    # expressions that has 'Operator' but actually does not represent
    # 'Operator' objects, e.g. an unevalauted Operator-to-Scalar map. These
    # objects can be grouped into 'NotAnOperator' provided by the package. 
    # Furthermore, the user may also use other noncommuting 'Expr's with the package, in
    # which case the implementation below will be confused because it expects
    # to get at least one 'sub' value. 
    if (all(not(item.has(Operator)) for item in seq)
        # or any(item.has(NotAnOperator) for item in seq) 
                # NOTE: Currently, 'NotAnOperator' is not implemented in the package, 
                # but we shall keep this here, commented, just in case.
        or any(not(isinstance(atom, Operator)) for item in seq 
               for atom in item.atoms() if getattr(atom, "is_commutative") is False)):
        return original_Mul_flatten(seq)
    
    ###
    
    c_part, nc_part, order_symbol = original_Mul_flatten(seq)
    # `nc_part`` contains our operators
    
    if not(nc_part):
        return c_part, nc_part, order_symbol
    
    # A universal Operator expression
    # is indicated by having at least one Operator with `has_sub=False`, 
    # We cannot reorder universally noncommuting expressions. As such,
    # they serve as a bound for the interval in 'nc_part' we can reoder
    # a time. We collect this interval into 'reorderable_nc' and reorder
    # it using Python's built-in 'sorted', by providing a sort key that 
    # checks whether a given entry can go before the previous entry based
    # on 'sub' in them. 
    
    reordered_nc_part = []
    reorderable_nc = []

    def treat_reorderable_nc() -> None:
    
        # NOTE: There may be better reordering rules we can implement
        # to make nonpolynomial expressions prettier, but the current
        # simple one already works very well for polynomials and does not
        # break for nonpolynomials.
        
        def can_move_left(A : sp.Expr, B : sp.Expr) -> bool:
            # Whether A can move to the left of B, assuming A is originally
            # to B's right. 
            
            A_sub = get_oper_sub(A)
            B_sub = get_oper_sub(B)
            
            # A can be safely reordered with respect to B if they do *not* share
            # any common 'sub's. 
                    
            if A_sub & B_sub: 
                return False
            
            # We determine whether A can move to the left of B following the sequence in which
            # the 'sub's appear in sub_cache. By doing it this way, we try our best to make the 'Operator'
            # ordering obey sympy's canonical ordering rule. This ordering rule is also *consistent*.
            
            A_sub_index_in_sub_cache = [sub_cache.index(sub) for sub in A_sub]
            B_sub_index_in_sub_cache = [sub_cache.index(sub) for sub in B_sub]
        
            # By construction, 'Operator's with 'has_sub=False' is not put into
            # 'reorderable_nc' and hence would not appear here, so no misleading
            # "empty 'sub'" from these objects. 
            #
            # Furthermore, these two lists must have at least one item. If the
            # list is empty, then the expression is noncommuting but is not 
            # the package's Operator. Something must have barged in here... considering
            # the filter above.
                        
            if min(A_sub_index_in_sub_cache) < min(B_sub_index_in_sub_cache):
                return True
            return False

        def cmp(A, B):
            if can_move_left(A, B):
                return -1
            elif can_move_left(B, A):
                return 1
            else:
                return 0
                
        reordered_nc_part.extend(list(sorted(reorderable_nc, 
                                             key=cmp_to_key(cmp))))
        
    for nc in nc_part:
        if not(is_universal(nc)):
            # As long as 'nc' is not universal (i.e., reorderable),
            # we can keep appending to reorderable_nc.
            reorderable_nc.append(nc)
        else:
            # A universal 'nc' must stay
            # to the right of all 'nc' to its left. As such, we can now
            # cut off the 'reorderable_nc' collection and reorder what's inside,
            # and empty the list afterwards. Then, we can add our universal 'nc'
            # of the current iteration.
            treat_reorderable_nc()
            reorderable_nc = []
            reordered_nc_part.append(nc)
    
    # There may be leftover 'nc' from the loop if the last one is not universal. 
    treat_reorderable_nc() 
                
    # We run it thorugh the original Mul.flatten again to let sympy clean
    # up the arguments, like merging adjacent factors into a Pow. The first
    # call does not do this because the two may be separated by another
    # object initially. 
    _, reordered_nc_part_reflattened, _ = original_Mul_flatten(reordered_nc_part)
    
    return c_part, reordered_nc_part_reflattened, order_symbol
