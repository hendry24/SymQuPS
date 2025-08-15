from .objects.base import Base
class s(Base):
    """
    The Clahill-Glauber s-parameter. Set its value by setting its `.val` attribute.
    """
    def _get_symbol_name_and_assumptions(cls, val):
        name = r"s"
        return name, {"commutative" : True}
        
    def __new__(cls):
        from sympy import sympify
        return super().__new__(cls, sympify(0))
        
    @property
    def val(self):
        if "_val" not in self.__dict__:
            return self._custom_args[0]
        else:
            return self._val

    @val.setter
    def val(self, value):
        accepted_types = (int, float, complex)
        if not isinstance(value, accepted_types):
            raise TypeError(f"Expected {accepted_types}, got {type(value)} instead.")
        from sympy import sympify
        self._val = sympify(value)
        
    def _latex(self, printer):
        return r"s = %s" % self.val
        
s = s()

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