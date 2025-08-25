import sympy as sp

import typing
from . import scalars 
# Avoid, e.g., "from scalars import hbar" 
# since we want its value to follow changes at runtime.
from .base import Base, qpTypePSO, alphaTypePSO
from .._internal.cache import sub_cache
from .._internal.basic_routines import treat_sub

# NOTE: 'import .._internal.operator_handling' will result
# in circular imports. 

class Operator(Base):
    
    base = NotImplemented
    has_sub = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        return r"%s_{%s}" % (cls.base, sub), {"commutative":False}
                                            # This should shut off the assumption system.
    
    def __new__(cls, sub = None):
        sub = treat_sub(sub, cls.has_sub)
        
        if cls.has_sub:
            sub_cache._append(sub)
        
        return super().__new__(cls, sub)
        
    @property
    def sub(self):
        return self._custom_args[0]
    
    def dagger(self):
        raise NotImplementedError()
    
    @property
    def _diff_wrt(self):
        msg = "No differentiation with respect to 'Operator'."
        raise NotImplementedError(msg)
    
class HermitianOp(Operator):
    @typing.final
    def dagger(self):
        return self

class qOp(HermitianOp, qpTypePSO):
    base = r"\hat{q}"
    
class pOp(HermitianOp, qpTypePSO):
    base = r"\hat{p}"
    
class annihilateOp(Operator, alphaTypePSO):
    base = r"\hat{a}"
        
    def define(self):
        with sp.evaluate(False):
            return ((qOp(sub=self.sub)*scalars.mu + sp.I*pOp(sub=self.sub)/scalars.mu) 
                    / sp.sqrt(2*scalars.hbar))
    
    def dagger(self):
        return createOp(sub = self.sub)
    
class createOp(Operator, alphaTypePSO):
    base = r"\hat{a}^{\dagger}"
    
    def define(self):
        mu_conj = sp.conjugate(scalars.mu)
        with sp.evaluate(False):
            return ((qOp(sub=self.sub)*mu_conj - sp.I*pOp(sub=self.sub)/mu_conj) 
                    / sp.sqrt(2*scalars.hbar))
        
    def dagger(self):
        return annihilateOp(sub = self.sub)
    
class densityOp(HermitianOp):
    base = r"\rho"
    has_sub = False
    
    def __new__(cls, sub=None):
        return super().__new__(cls, sub)
    
global rho
rho = densityOp()