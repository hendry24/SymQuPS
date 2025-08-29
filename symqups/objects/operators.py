import sympy as sp
import typing

from .base import Base
from .._internal.grouping import HilbertSpaceObject, qpType, alphaType, UnDualBoppable
from .._internal.cache import sub_cache
from .._internal.basic_routines import treat_sub

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
        msg = "No differentiation with respect to 'Operator'."
        raise NotImplementedError(msg)
    
class HermitianOp(Operator):
    @typing.final
    def dagger(self):
        return self

class qOp(HermitianOp, qpType):
    base = r"\hat{q}"
    
class pOp(HermitianOp, qpType):
    base = r"\hat{p}"
    
class annihilateOp(Operator, alphaType):
    base = r"\hat{a}"
    
    def dagger(self):
        return createOp(sub = self.sub)
    
class createOp(Operator, alphaType):
    base = r"\hat{a}^{\dagger}"
        
    def dagger(self):
        return annihilateOp(sub = self.sub)

class _CommutatorSymbol(Base, HilbertSpaceObject):
    def _get_symbol_name_and_assumptions(cls, left):
        return r"\left[%s, \cdot\right]" % (sp.latex(left)), {"commutative" : False}

    def __new__(cls, left : sp.Expr):
        return super().__new__(cls, left)
    
    @property
    def left(self):
        return self._custom_args[0]


class densityOp(HermitianOp, UnDualBoppable):
    base = r"\rho"
    has_sub = False
    
    def __new__(cls, sub=None):
        return super().__new__(cls, sub)
    
global rho
rho = densityOp()