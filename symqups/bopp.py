import sympy as sp

from ._internal.grouping import (
    HilbertSpaceObject, PhaseSpaceObject, NotAnOperator, Acting,
    NotAScalar
)
from ._internal.preprocessing import preprocess_class

from .objects.base import Base
from .objects.operators import annihilateOp, createOp

from .manipulations import dagger

from . import s as CahillGlauberS

@preprocess_class
class _BoppActor(Base, Acting):
    Hilbert : bool
    
    def _get_symbol_name_and_assumptions(cls, base, target, left):
        dir = "L" if left else "R"
        return r"\hat{\mathcal{B}}_{%s}^{%s}" % (sp.latex(base), dir), {"commutative":True}
    
    def __new__(cls,
                base : annihilateOp|createOp, 
                target : sp.Expr|None = None,
                left : bool = False):
        obj = super().__new__(cls, base, target, left)
        obj._base = base
        obj._left = left
        
        if target is None:
            return obj
        
        return obj.act(target)
    
    @property
    def base(self):
        return self._base
    
    @property
    def left(self):
        return self._left
    
    def act(self, target : sp.Expr):
        s = CahillGlauberS.val
        sgn = -1 if isinstance(self.base, annihilateOp) else 1
        space_sgn = -1 if self.Hilbert else 1
        if self.left:
            return target*self.base + space_sgn * sp.Rational(1,2)*(s+sgn)*sp.Derivative(target, dagger(self.base))
        else:
            return self.base*target + space_sgn * sp.Rational(1,2)*(s-sgn)*sp.Derivative(target, dagger(self.base))

class HilbertSpaceBoppSuperoperator(_BoppActor, HilbertSpaceObject, NotAnOperator):
    Hilbert = True
HSBS = HilbertSpaceBoppSuperoperator
    
class PhaseSpaceBoppOperator(_BoppActor, PhaseSpaceObject, NotAScalar):
    Hilbert = False
PSBO = PhaseSpaceBoppOperator