import sympy as sp
from sympy.core.sympify import CantSympify
import warnings

from .grouping import _ReadOnlyExpr
from ..objects.base import Base

class Constant(CantSympify, _ReadOnlyExpr, Base):
    """
    Base class for package parameters. Set its value by setting the `.val`
    attribute.
    """
    name = "Parameter"
    default_value = NotImplemented
    
    def _get_symbol_name_and_assumptions(cls, val):
        return cls.name, {"commutative" : True}
        
    def __new__(cls):
        return super().__new__(cls, cls.default_value)
        
    @property
    def val(self):
        if "_val" not in self.__dict__:
            return self._custom_args[0]
        else:
            return self._val

    @val.setter
    def val(self, value):
        self._val = sp.sympify(value)
        
        from .cache import sub_cache
        sub_cache._refresh_all()
        
    def _latex(self, printer):
        return r"%s = %s" % (self.name, sp.latex(self.val))
    
class CahillGlauberSParameter(Constant):
    name = "Cahill-Glauber s parameter"
    default_value = sp.Number(0)
        
    @Constant.val.setter
    def val(self, value):
        value = sp.sympify(value)
        if ((isinstance(value, sp.Number) 
             and value.is_real
             and value <= 1 
             and value >= -1)
            or isinstance(value, sp.Symbol)):
            pass
        else:
            msg = "The Cahill-Glauber formalism works best when 's' is real and lies between -1 and 1. "
            msg += "The input value does not identify as such, nor is it a 'sympy.Symbol'."
            warnings.warn(msg)
        super(CahillGlauberSParameter,
              CahillGlauberSParameter).val.__set__(self, value)
            
    def _latex(self, printer):
        return r"\text{%s,}\quad s = %s" % (self.name, sp.latex(self.val))
    
class ReducedPlanckConstant(Constant):
    name = r"\hbar"
    default_value = sp.Symbol(r"hbar")
    
class AlphaScalingParameter(Constant):
    name = r"\zeta"
    default_value = sp.Symbol(r"zeta")
    
class piTranscendentalNumber(Constant):
    name = r"\pi"
    default_value = sp.Symbol(r"pi")
    
    @Constant.val.setter
    def val(self, value):
        raise NotImplementedError("Cannot set the value of 'pi'.")