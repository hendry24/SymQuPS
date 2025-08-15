import sympy as sp

from .base import Base
from .cache import _sub_cache
from ..utils._internal_routines import _treat_sub, _screen_type

__all__ = ["q", "p", "alpha", "alphaD", "W"]

global hbar, pi, mu
hbar = sp.Symbol(r"hbar", real=True, positive=True)
pi = sp.Symbol(r"pi", real=True, positive=True)
mu = sp.Symbol(r"mu", real=False)

class Scalar(Base):
    base = NotImplemented
    has_sub = True
    is_real = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        name = r"%s_{%s}" % (cls.base, sub)
        return name, {"real" : cls.is_real}
        
    def __new__(cls, sub = None):
        sub = _treat_sub(sub, cls.has_sub)
        
        global _sub_cache
        _sub_cache._update([sub])

        return super().__new__(cls, sub)
        
    @property
    def sub(self):
        return self._custom_args[0]
    
class t(Scalar):
    base = r"t"
    has_sub = False
    
class q(Scalar):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"q"
    
class p(Scalar):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"p"
    
class alpha(Scalar):
    base = r"\alpha"
    is_real = False
    
    def define(self):
        with sp.evaluate(False):
            return (mu*q(self.sub) + sp.I * p(self.sub) / mu) / sp.sqrt(2*hbar)
    
    def conjugate(self):
        return alphaD(self.sub)
    def _eval_conjugate(self):
        return self.conjugate()
    
class alphaD(Scalar):
    base = r"{\alpha^*}"
    is_real = False
    
    def define(self):
        with sp.evaluate(False):
            mu_conj = sp.conjugate(mu)
            return (mu_conj*q(self.sub) - sp.I * p(self.sub) / mu_conj) / sp.sqrt(2*hbar)
        
    def conjugate(self):
        return alpha(self.sub)
    def _eval_conjugate(self):
        return self.conjugate()
###

class _Primed(Base):
    def _get_symbol_name_and_assumptions(cls, A):
        return r"{%s}'" % sp.latex(A), {"commutative" : False}
    
    def __new__(cls, A : sp.Expr):
        
        A = sp.sympify(A)
        
        if isinstance(A, (q, p, alpha, alphaD)):
            return super().__new__(cls, A)
        
        return A.subs({X:_Primed(X) for X in A.atoms(q,p,alpha,alphaD)})
    
    @property
    def base(self):
        return self._custom_args[0]
    
class _DePrimed():
    def __new__(cls, A : sp.Expr):
        subs_dict = {X : X.base for X in A.atoms(_Primed)}
        return A.subs(subs_dict)

###

class _DerivativeSymbol(Base):
    
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

####

class StateFunction(sp.Function):
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

    show_vars = False
    
    def _latex(self, printer):
        if self.show_vars:
            return str(self).replace("StateFunction", r"W_s")
        return r"W_s"
    
class W():
    """
    The 'StateFunction' constructor. Constructs 'StateFunction' using cached 'q' and 'p' as 
    variables. This is the recommended way to create the object since a user might miss 
    some variables with manual construction, leading to incorrect evaluations.
    """
    def __new__(cls, show_vars=False):
        global _sub_cache
        vars = []
        for sub in _sub_cache:
            vars.extend([q(sub), p(sub)])
        
        obj : StateFunction = StateFunction(t(), *vars)
        obj.show_vars = show_vars
        
        """
        Instantiating two StateFunction objects would return the SAME
        function thanks to SymPy's caching. As such, the fact that
        the latest instantiation always overrides show_vars is not a problem.
        """
        
        return obj