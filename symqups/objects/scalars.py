import sympy as sp

from .. import s as CahillGlauberS
from .base import Base
from .._internal.grouping import (qpType, alphaType, PhaseSpaceVariable, 
                                  PhaseSpaceObject, UnBoppable, PrimedPSO)
from .._internal.cache import sub_cache
from .._internal.basic_routines import treat_sub, invalid_input

__all__ = ["q", "p", "alpha", "alphaD", "W"]

class Scalar(Base):
    base = NotImplemented
    has_sub = True
    is_real = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        name = r"%s_{%s}" % (cls.base, sub)
        return name, {"real" : cls.is_real, "commutative" : True}
        
    def __new__(cls, sub = None):
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
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"q"
    
class p(Scalar, PhaseSpaceVariable, qpType):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"p"
    
class alpha(Scalar, PhaseSpaceVariable, alphaType):
    base = r"\alpha"
    is_real = False
    
    def conjugate(self):
        return alphaD(self.sub)
    def _eval_conjugate(self):
        return self.conjugate()
    
class alphaD(Scalar, PhaseSpaceVariable, alphaType):
    base = r"\overline{\alpha}"
    is_real = False
        
    def conjugate(self):
        return alpha(self.sub)
    def _eval_conjugate(self):
        return self.conjugate()
###

class _Primed(Base, PrimedPSO):
    def _get_symbol_name_and_assumptions(cls, A):
        return r"{%s}'" % sp.latex(A), {"commutative" : False}
    
    def __new__(cls, A : sp.Expr):
        A = sp.sympify(A)
        
        if isinstance(A, PhaseSpaceVariable):
            return super().__new__(cls, A)
        
        return A.subs({X:_Primed(X) for X in A.atoms(PhaseSpaceVariable)})
    
    @property
    def base(self):
        return self._custom_args[0]

###

class _DerivativeSymbol(Base, PrimedPSO):
    
    def _get_symbol_name_and_assumptions(cls, primed_phase_space_coordinate : _Primed):
        return r"\frac{\partial \cdot}{\partial {%s}}" % sp.latex(primed_phase_space_coordinate.base), {"commutative":False}
    
    def __new__(cls, primed_phase_space_coordinate : _Primed):
        if not(isinstance(primed_phase_space_coordinate, _Primed)):
            raise ValueError(r"'_DifferentialSymbol' expects '_Primed', but got '%s' instead" % \
                type(primed_phase_space_coordinate))
            
        return super().__new__(cls, primed_phase_space_coordinate)
    
    @property
    def diff_var(self):
        return self._custom_args[0]

###

class StateFunction(sp.Expr, PhaseSpaceObject, UnBoppable):
    """
    The state function object.
    
    Parameters
    ----------
        
    *vars
        Variables of the Wigner function. 
    
    s : complex number
        s-paremeter defined within the Cahill-Glauber formalism. Some special
        values of s:
            - `s = 0` : Wigner function
            - `s = 1` : Glauber P function
            - `s = -1` : Husimi Q function
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
                return r"W_{s=%s}" % sp.latex(CahillGlauberS.val)
    
global W
W = StateFunction(t())
