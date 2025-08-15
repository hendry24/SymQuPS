import sympy as sp

from .base import Base, _sub_cache, _treat_sub

__all__ = ["q", "p", "alpha", "alphaD", "W"]

global hbar, pi
hbar = sp.Symbol(r"hbar", real=True)
pi = sp.Symbol(r"pi", real=True)

class Scalar(Base):
    base = NotImplemented
    has_sub = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        name = r"%s_{%s}" % (cls.base, sub)
        return name, {"real" : True}
        
    def __new__(cls, sub = None):
        sub = _treat_sub(sub, cls.has_sub)
        
        global _sub_cache
        _sub_cache._update([sub])

        return super().__new__(cls, sub)
        
    @property
    def sub(self):
        return self._custom_args[0]
    
    def weyl_transform(self):
        raise NotImplementedError()
    
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
    
    def weyl_transform(self):
        from .hilbert_operators import qOp
        return qOp(self.sub)
    
class p(Scalar):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"p"
    
    def weyl_transform(self):
        from .hilbert_operators import pOp
        return pOp(self.sub)
    
class alpha():
    def __new__(cls, sub = None):
        with sp.evaluate(False):
            return (1 / sp.sqrt(2*hbar)) * (q(sub) + sp.I * p(sub))
        
class alphaD():
    def __new__(cls, sub = None):
        with sp.evaluate(False):
            return (1 / sp.sqrt(2*hbar)) * (q(sub) - sp.I * p(sub))

###

class _Primed(Base):
    def _get_symbol_name_and_assumptions(cls, A):
        return r"{%s}'" % (sp.latex(A)), {"commutative" : False}
    
    def __new__(cls, A : sp.Expr):
        
        A = sp.sympify(A)
        
        if isinstance(A, (q,p)):
            return super().__new__(cls, A)
        
        return A.subs({X:_Primed(X) for X in A.atoms(q,p)})
    
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
        return r"\partial_{%s}" % (sp.latex(primed_phase_space_coordinate)), {"commutative":False}
    
    def __new__(cls, primed_phase_space_coordinate):
        if not(isinstance(primed_phase_space_coordinate, _Primed)):
            raise ValueError(r"'_DifferentialSymbol' expects '_Primed', but got '%s' instead" % \
                type(primed_phase_space_coordinate))
            
        return super().__new__(cls, primed_phase_space_coordinate)
    
    @property
    def diff_var(self):
        return self._custom_args[0]

####

class WignerFunction(sp.Function):
    show_vars = False
    """
    The Wigner function object.
    
    Parameters
    ----------
    
    *vars
        Variables of the Wigner function. 
    
    """
    def _latex(self, printer):
        if self.show_vars:
            return str(self).replace("WignerFunction", "W")
        return r"W"
    
    def weyl_transform(self):
        from .hilbert_operators import rho
        global _sub_cache, pi, hbar
        N = len(_sub_cache)
        return rho() / (2*pi*hbar)**N
    
class W():
    """
    The 'WignerFunction' constructor. Constructs 'WignerFunction' using cached 'q' and 'p' as 
    variables. This is the recommended way to create the object since a user might miss 
    some variables with manual construction, leading to incorrect evaluations.
    """
    def __new__(cls):
        global _sub_cache
        vars = []        
        for sub in _sub_cache:
            vars.extend([q(sub), p(sub)])
        
        return WignerFunction(t(), *vars)