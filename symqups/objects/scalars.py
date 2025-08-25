import sympy as sp

from .base import Base
from .._internal.grouping import qpType, alphaType, PhaseSpaceObject, UnBoppable, PrimedPSO
from .._internal.cache import sub_cache
from .._internal.basic_routines import treat_sub, invalid_input

__all__ = ["q", "p", "alpha", "alphaD", "W"]

global hbar, pi, mu
hbar = sp.Symbol(r"hbar", real=True, positive=True)
pi = sp.Symbol(r"pi", real=True, positive=True)
mu = sp.Symbol(r"mu", real=True, positive=True)

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

class q(Scalar, PhaseSpaceObject, qpType):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"q"
    
class p(Scalar, PhaseSpaceObject, qpType):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"p"
    
class alpha(Scalar, PhaseSpaceObject, alphaType):
    base = r"\alpha"
    is_real = False
    
    def define(self):
        with sp.evaluate(False):
            return (mu*q(self.sub) + sp.I * p(self.sub) / mu) / sp.sqrt(2*hbar)
    
    def conjugate(self):
        return alphaD(self.sub)
    def _eval_conjugate(self):
        return self.conjugate()
    
class alphaD(Scalar, PhaseSpaceObject, alphaType):
    base = r"\overline{\alpha}"
    is_real = False
    
    def define(self):
        mu_conj = sp.conjugate(mu)
        with sp.evaluate(False):
            return (mu_conj*q(self.sub) - sp.I * p(self.sub) / mu_conj) / sp.sqrt(2*hbar)
        
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
        
        if isinstance(A, PhaseSpaceObject):
            return super().__new__(cls, A)
        
        return A.subs({X:_Primed(X) for X in A.atoms(PhaseSpaceObject)})
    
    @property
    def base(self):
        return self._custom_args[0]
    
def _deprime(expr : sp.Expr):
    subs_dict = {X : X.base for X in expr.atoms(_Primed)}
    return expr.subs(subs_dict)

###

class _DerivativeSymbol(Base, PrimedPSO):
    
    def _get_symbol_name_and_assumptions(cls, primed_phase_space_coordinate):
        return r"\partial_{%s}" % sp.latex(primed_phase_space_coordinate), {"commutative":False}
    
    def __new__(cls, primed_phase_space_coordinate):
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
        return r"W_s"
    
global W
W = StateFunction(t())

# In the previous version, W retrieves 'sub_cache' to use as its arguments
# at instantiation. However, this is problematic when W is instantiated
# before other objects it is supposed to interact with, for example in
# `Star(alpha(1), W(), alpha(2))`. Here W() only sees the sub '1' when
# it is constructed. A better construction would to let the variable
# W be updated each time 'sub_cache' is updated. See
# `cache.sub_cache.update`.
