import sympy as sp
from sympy.core.sympify import CantSympify
import warnings

from .._internal.grouping import _ReadOnlyExpr
from .._internal.preprocessing import preprocess_func

from .base import Base

class Constant(CantSympify, _ReadOnlyExpr, Base):
    """
    Base class for package constants. Set its value by setting the `.val`
    attribute.
    """
    name = "Constant"
    default_value = NotImplemented
    
    def _get_symbol_name_and_assumptions(cls):
        return cls.name, {"commutative" : True}
        
    def __new__(cls):
        obj = super().__new__(cls)
        obj._val = cls.default_value
        return obj
        
    @property
    def val(self):
        return self._val

    @val.setter
    @preprocess_func
    def val(self, value):        
        self._val = value
        
        from .._internal.cache import sub_cache
        sub_cache._refresh_all()
        
    def _latex(self, printer):
        return r"%s = %s" % (self.name, sp.latex(self.val))
    
class CahillGlauberSParameter(Constant):
    name = "Cahill-Glauber s parameter"
    default_value = sp.Symbol("s")
        
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

class PositiveRealConstant(Constant):
    
    @Constant.val.setter
    def val(self, value):
        value = sp.sympify(value)
        if isinstance(value, sp.Symbol):
            if value.is_real is None:
                value = sp.Symbol(value.name, real=True)
            elif value.is_real is False:
                msg = f"Symbolic value for {self.__class__.__name__} must be real."
                raise AttributeError(msg)
            
        elif not(hasattr(value, "is_positive")) or not(value.is_positive):
            msg = f"Invalid value for {self.__class__.__name__}. Value must be a real 'Symbol' or "
            msg += f"a positive number."
            raise AttributeError(msg)
        
        super(PositiveRealConstant,
              PositiveRealConstant).val.__set__(self, value)

###

class ReducedPlanckConstant(PositiveRealConstant):
    name = r"\hbar"
    default_value = sp.Symbol(r"hbar")
    
class AlphaScalingParameter(PositiveRealConstant):
    name = r"\zeta"
    default_value = sp.Symbol(r"zeta")