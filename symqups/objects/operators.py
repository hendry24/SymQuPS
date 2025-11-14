import typing

from .base import Base
from .._internal.grouping import HilbertSpaceObject, qpType, alphaType, PhaseSpaceVariableOperator, CannotBoppShift
from .._internal.cache import sub_cache
from .._internal.basic_routines import treat_sub

# NOTE: 'import .._internal.operator_handling' will result
# in circular imports. 

class Operator(Base, HilbertSpaceObject):
    """
    Base class for Hilbert space operators.
    """
    
    base = NotImplemented
    has_sub = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        return r"%s_{%s}" % (cls.base, sub), {"commutative" : False}
                                            # This should shut off the assumption system.
    
    def __new__(cls, sub = None):
        """
        Construct an operator object.
        
        Parameters
        ----------

        sub : sympify-able, default: None
            Subscript of the operator representing the subsystem. If `None`,
            then `sympy.Symbol("")` is used to represent no subscript.
            
        """
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
        msg = "Differentiation with respect to 'Operator' is undefined for SymPy. "
        msg += "This package, however, implements the formal derivative with respect to"
        msg += "'annihilateOp' and 'createOp'. See this package's 'Derivative' object."
        raise NotImplementedError(msg)
    
###

class HermitianOp(Operator):
    """
    Base class for hermitian operators.
    """
    @typing.final
    def dagger(self):
        return self
    
###

class qOp(HermitianOp, qpType, PhaseSpaceVariableOperator):
    """
    The operator corresponding to the canonical position `q`.
    """
    base = r"\hat{q}"
    
class pOp(HermitianOp, qpType, PhaseSpaceVariableOperator):
    """
    The operator corresponding to the canonical momentum `p`.
    """
    base = r"\hat{p}"

###
 
class annihilateOp(Operator, alphaType, PhaseSpaceVariableOperator):
    """
    The annihilation operator. 
    """
    base = r"\hat{a}"
    
    def dagger(self):
        return createOp(sub = self.sub)
        
class createOp(Operator, alphaType, PhaseSpaceVariableOperator):
    """
    The creation operator.
    """
    base = r"\hat{a}^{\dagger}"
        
    def dagger(self):
        return annihilateOp(sub = self.sub)
        
###

class densityOp(HermitianOp, CannotBoppShift):
    """
    The density operator or density matrix.
    """
    base = r"\rho"
    has_sub = False
    
    def __new__(cls, _sub = None):
        return super().__new__(cls, _sub)
    
global rho
rho = densityOp()