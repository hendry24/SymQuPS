import sympy as sp

from .. import s as CahillGlauberS
from .base import Base
from .._internal.grouping import (qpType, alphaType, PhaseSpaceVariable, 
                                  PhaseSpaceObject, CannotBoppShift)
from .._internal.cache import sub_cache
from .._internal.basic_routines import treat_sub, invalid_input

__all__ = ["q", "p", "alpha", "alphaD", "W"]

class Scalar(Base):
    """
    Base class for scalar objects.
    """
    base = NotImplemented
    has_sub = True
    is_real = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        name = r"%s_{%s}" % (cls.base, sub)
        return name, {"real" : cls.is_real, "commutative" : True}
        
    def __new__(cls, sub = None):
        """
        Construct an operator object.
        
        Parameters
        ----------

        sub : sympify-able, default: None
            Subscript of the scalar representing the subsystem. If `None`,
            then `sympy.Symbol("")` is used to represent no subscript.
        
        """
 
        sub = treat_sub(sub, cls.has_sub)
        
        if cls.has_sub:
            sub_cache._append(sub)

        return super().__new__(cls, sub)
        
    @property
    def sub(self):
        return self._custom_args[0]
    
###

class t(Scalar):
    base = r"t"
    has_sub = False

###

class q(Scalar, PhaseSpaceVariable, qpType):
    """
    The canonical position operator or first phase-space quadrature.
    """
    base = r"q"
    
class p(Scalar, PhaseSpaceVariable, qpType):
    """
    The canonical momentum operator or second phase-space quadrature.
    """
    base = r"p"
    
class alpha(Scalar, PhaseSpaceVariable, alphaType):
    """
    The complex phase space amplitude.
    """
    base = r"\alpha"
    is_real = False
    
    def conjugate(self):
        return alphaD(self.sub)
    def _eval_conjugate(self):
        return self.conjugate()
    
class alphaD(Scalar, PhaseSpaceVariable, alphaType):
    """
    The formal variable representing the complex conjugate of `alpha`.
    """
    base = r"\overline{\alpha}"
    is_real = False
        
    def conjugate(self):
        return alpha(self.sub)
    
    def _eval_conjugate(self):
        return self.conjugate()

###

class StateFunction(sp.Expr, PhaseSpaceObject, CannotBoppShift):
    """
    The state function object.
    
    Parameters
    ----------
        
    *vars
        Variables of the state function. 
    """
    
    @property
    def args(self):
        return self._args
    
    def _set_args(self, value):
        if not(all(isinstance(x, (t, alpha, alphaD))
                   for x in value)):
            invalid_input(value, "StateFunction.args")
        
        self._args = value
    
    ###
    
    def _latex(self, printer):
        match CahillGlauberS.val:
            case -1:
                return r"Q"
            case 0:
                return r"W"
            case 1:
                return r"P"
            case default:
                return r"W_{s}"
    
global W
W = StateFunction(t())
