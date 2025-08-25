import sympy as sp
from sympy.core.sympify import CantSympify

from .grouping import _ReadOnlyExpr
from ..objects.base import Base

class CahillGlauberSParameter(CantSympify, _ReadOnlyExpr, Base):
    """
    The Clahill-Glauber s-parameter. Set its value by setting its `.val` attribute.
    """
    def _get_symbol_name_and_assumptions(cls, val):
        name = r"s"
        return name, {"commutative" : True}
        
    def __new__(cls):
        return super().__new__(cls, sp.Number(0))
        
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