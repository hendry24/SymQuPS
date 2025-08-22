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

del CahillGlauberSParameter, Base, _ReadOnlyExpr, CantSympify

###

_original_Mul_flatten = sp.Mul.flatten

class sMul(sp.Mul):
    """
    `sympy`'s original Mul.
    """
    pass
sMul.flatten = _original_Mul_flatten

del _original_Mul_flatten

###

from .utils._internal._operator_handling import _get_oper_sub
from .objects.cache import _sub_cache

def _patched_Mul_flatten(seq, 
                         _get_oper_sub = _get_oper_sub,
                         _sub_cache = _sub_cache):  
                        # We save the references here so that
                        # their names can be deleted and not clutter
                        # the import. 
    c_part, nc_part, order_symbol = sMul.flatten(seq)
    
    # nc_part contains our operators
    
    nc_part_sub_lst = [_get_oper_sub(A) for A in nc_part]
    
    used_nc_part_idx = []
    reordered_nc_part = [] # since two arguments may be the same object.
    for sub in _sub_cache:
        
        for j, arg_sub in enumerate(nc_part_sub_lst):
    
            if ((not(arg_sub) or (sub in arg_sub))
                and j not in used_nc_part_idx):
    
                reordered_nc_part.append(nc_part[j])
                used_nc_part_idx.append(j)
    
    return c_part, reordered_nc_part, order_symbol

sp.Mul.flatten = _patched_Mul_flatten

del _patched_Mul_flatten, _get_oper_sub, _sub_cache, sp

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