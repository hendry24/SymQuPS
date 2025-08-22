import sympy as sp
from sympy.core.sympify import CantSympify

from .objects.base import Base, _ReadOnlyExpr

class CahillGlauberSParameter(CantSympify, _ReadOnlyExpr, Base):
    """
    The Clahill-Glauber s-parameter. Set its value by setting its `.val` attribute.
    """
    def _get_symbol_name_and_assumptions(cls, val):
        name = r"s"
        return name, {"commutative" : True}
        
    def __new__(cls):
        from sympy import Number
        return super().__new__(cls, Number(0))
        
    @property
    def val(self):
        if "_val" not in self.__dict__:
            return self._custom_args[0]
        else:
            return self._val

    @val.setter
    def val(self, value):
        from sympy import sympify, Number
        value = sympify(value)
        if (isinstance(value, Number) 
            and value.is_real
            and value <= 1 
            and value >= -1):
            pass
        else:
            from pprint import pprint
            msg = "The Cahill-Glauber formalism works best when 's' is real and lies between -1 and 1. "
            msg += "The input value does not identify as such."
            pprint(msg)
        self._val = value
            
    def _latex(self, printer):
        from sympy import latex
        return r"\text{Cahill-Glauber s parameter,}\quad s = %s" % latex(self.val)
    
s = CahillGlauberSParameter()

del CantSympify, Base, _ReadOnlyExpr, CahillGlauberSParameter

###

from .utils._internal._operator_handling import _get_oper_sub
from .objects.cache import _sub_cache
from .objects.operators import Operator

def _patched_Mul_flatten(seq, 
                         _get_oper_sub = _get_oper_sub,
                         _sub_cache = _sub_cache,
                         _original_Mul_flatten = sp.Mul.flatten,
                         Operator = Operator): 
                        # Specifying default arguments saves the names
                        # into the function.
    c_part, nc_part, order_symbol = _original_Mul_flatten(seq)
    # This automatically flattens Mul input into another Mul.
    # `nc_part`` contains our operators
        
    # A universal Operator expression
    # is indicated by having at least one Operator with `has_sub=False`, 
    # We cannot reorder universally noncommuting expressions. 

    def _is_universal(A : sp.Expr) -> bool:
        return not(all(atom.has_sub for atom in A.atoms(Operator)))

    reordered_nc_part = []
    reorderable_nc = []

    def _treat_reorderable_nc():
        nc_sub_lst = [_get_oper_sub(nc) for nc in reorderable_nc]

        used_nc_idx = []
        for sub in _sub_cache:
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
        if not(_is_universal(nc)):
            # As long as 'nc' is not universal (i.e., reorderable),
            # we can keep appending to reorderable_nc.
            reorderable_nc.append(nc)
        else:
            # A universal 'nc' must stay
            # to the right of all 'nc' to its left. As such, we can now
            # cut off the 'reorderable_nc' collection and reorder what's inside,
            # and empty the list afterwards. Then, we can add our universal 'nc'
            # of the current iteration.
            _treat_reorderable_nc()
            reorderable_nc = []
            reordered_nc_part.append(nc)
    
    # There may be leftover 'nc' from the loop if the last one is not universal. 
    _treat_reorderable_nc() 
                
    # We run it thorugh the original Mul.flatten again to let sympy clean
    # up the arguments, like merging adjacent factors into a Pow. The first
    # call does not do this because the two may be separated by another
    # object initially. 
    _, reordered_nc_part, _ = _original_Mul_flatten(reordered_nc_part)
    
    return c_part, reordered_nc_part, order_symbol

sp.Mul.flatten = _patched_Mul_flatten

del _get_oper_sub, _sub_cache, Operator, _patched_Mul_flatten

###

# from .objects.scalars import q, p, alpha, alphaD, P
# from .objects.hilbert_operators import (qOp, pOp, 
#                                createOp, annihilateOp, 
#                                Dagger, rho)

# from .transforms.star_product import Bopp, Star

# from .transforms.wigner_transform import WignerTransform
# from .keep.eom import LindbladMasterEquation

# from .utils.multiprocessing import MP_CONFIG
# from .utils.grouping import collect_by_derivative, derivative_not_in_num

# from .objects.scalars import *