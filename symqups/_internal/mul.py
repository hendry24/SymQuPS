import sympy as sp

global original_Mul_flatten
original_Mul_flatten = sp.Mul.flatten

def patched_Mul_flatten(seq): 
    from .operator_handling import get_oper_sub
    from .cache import sub_cache
    from ..objects.operators import Operator

    c_part, nc_part, order_symbol = original_Mul_flatten(seq)
    # This automatically flattens Mul input into another Mul.
    # `nc_part`` contains our operators
        
    # A universal Operator expression
    # is indicated by having at least one Operator with `has_sub=False`, 
    # We cannot reorder universally noncommuting expressions. 

    def is_universal(A : sp.Expr) -> bool:
        return not(all(atom.has_sub for atom in A.atoms(Operator)))

    reordered_nc_part = []
    reorderable_nc = []

    def treat_reorderable_nc():
        nc_sub_lst = [get_oper_sub(nc) for nc in reorderable_nc]

        used_nc_idx = []
        for sub in sub_cache:
            # Since _sub_cache (as well as Scalar) is ordered according to
            # sympy's canon, we try to do the same for Operator, the ordering
            # of which sympy does not automatically do due to the noncommuting
            # nature.
            
            for j, nc_sub in enumerate(nc_sub_lst):
                
                if (sub in nc_sub) and (j not in used_nc_idx):
                    
                    reordered_nc_part.append(reorderable_nc[j])
                    used_nc_idx.append(j)
                    
        assert len(used_nc_idx) == len(reorderable_nc)

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
    _, reordered_nc_part, _ = original_Mul_flatten(reordered_nc_part)
    
    return c_part, reordered_nc_part, order_symbol
