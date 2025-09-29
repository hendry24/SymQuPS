import typing
import sympy as sp

from .base import Base
from .._internal.grouping import HilbertSpaceObject, qpType, alphaType, PhaseSpaceVariableOperator, CannotBoppShift
from .._internal.cache import sub_cache
from .._internal.basic_routines import treat_sub

from .scalars import t

# NOTE: 'import .._internal.operator_handling' will result
# in circular imports. 

class Operator(Base, HilbertSpaceObject):
    
    base = NotImplemented
    has_sub = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        return r"%s_{%s}" % (cls.base, sub), {"commutative" : False}
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
        msg = "Differentiation with respect to this 'Operator' is undefined."
        raise NotImplementedError(msg)
    
    def diff(self, *symbols, **assumptions):
        return sp.Derivative(self, *symbols, **assumptions)

###

class HermitianOp(Operator):
    @typing.final
    def dagger(self):
        return self
    
###

class qOp(HermitianOp, qpType, PhaseSpaceVariableOperator):
    base = r"\hat{q}"
    
class pOp(HermitianOp, qpType, PhaseSpaceVariableOperator):
    base = r"\hat{p}"

###
 
class annihilateOp(Operator, alphaType, PhaseSpaceVariableOperator):
    base = r"\hat{a}"
    
    def dagger(self):
        return createOp(sub = self.sub)
    
    @property
    def _diff_wrt(self):
        return True
    
class createOp(Operator, alphaType, PhaseSpaceVariableOperator):
    base = r"\hat{a}^{\dagger}"
        
    def dagger(self):
        return annihilateOp(sub = self.sub)
    
    @property
    def _diff_wrt(self):
        return True
    
###

class densityOp(HermitianOp, CannotBoppShift):
    base = r"\rho"
    has_sub = False
    
    def __new__(cls, _sub=None):
        return super().__new__(cls, _sub)
    
global rho
rho = densityOp()