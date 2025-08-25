import sympy as sp
from typing import Tuple, Sequence
from functools import cmp_to_key

global original_Mul_flatten
original_Mul_flatten = sp.Mul.flatten

# HACK: Monkey patching sympy's core implementation to do something
# extra if 'Operator' objects are input.
def patched_Mul_flatten(seq : Sequence) -> Tuple[list, list, list]: 
    from .operator_handling import get_oper_sub
    from .cache import sub_cache
    from .operator_handling import is_universal
    from ..objects.operators import Operator

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
    
    if all(not(item.has(Operator)) for item in seq):
        return original_Mul_flatten(seq)
    
    c_part, nc_part, order_symbol = original_Mul_flatten(seq)
    # `nc_part`` contains our operators
    
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
            
            first_A_sub_in_sub_cache = min(sub_cache.index(sub) for sub in A_sub)
            first_B_sub_in_sub_cache = min(sub_cache.index(sub) for sub in B_sub)

            if first_A_sub_in_sub_cache < first_B_sub_in_sub_cache:
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
