import sympy as sp

from ._internal.grouping import (
    HilbertSpaceObject, PhaseSpaceObject, NotAnOperator, Acting,
    NotAScalar
)
from ._internal.preprocessing import preprocess_class

from .objects.base import Base
from .objects.scalars import alpha, alphaD
from .objects.operators import annihilateOp, createOp

from .manipulations import dagger, Derivative

from . import s as CahillGlauberS

@preprocess_class
class _BoppActor(Base, Acting):
    """
    Base class for HSBS and PSBO.
    """
    Hilbert : bool
    
    def _get_symbol_name_and_assumptions(cls, base, target, left):
        dir = "L" if left else "R"
        return r"\hat{\mathcal{B}}_{%s}^{%s}" % (sp.latex(base), dir), {"commutative":True}
    
    def __new__(cls,
                base : annihilateOp|createOp|alpha|alphaD, 
                target : sp.Expr|None = None,
                left : bool = False):
        """
        Construct a HSBS or a PSBO. For the formalism, see Appendix B in 
        https://arxiv.org/abs/2509.17106.
        
        Parameters
        ----------
        
        base : annihilateOp or createOp or alpha or alphaD
            Base for the HSBS or PSBO. For example, a PSBO with base `alpha`
            is `alpha + (s + lr_sign)/2 * d_(alphaD)`, where `lr_sign` depends
            on the operation direction.
            
        target : sp.Expr, optional
            Target of the HSBS or PSBO. If not specified, then the constructor 
            returns a HSBS or PSBO object that cannot form a `sympy.Mul`. The
            user must first make it act on a target using the `.act` method.
            
        left : bool, default = False
            Whether the HSBS or PSBO acts to its left.
        """
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
        sgn = -1 if isinstance(self.base, (annihilateOp, alpha)) else 1
        space_sgn = -1 if self.Hilbert else 1
        if self.left:
            return target*self.base + space_sgn * sp.Rational(1,2)*(s+sgn)*Derivative(target, dagger(self.base))
        else:
            return self.base*target + space_sgn * sp.Rational(1,2)*(s-sgn)*Derivative(target, dagger(self.base))

class HilbertSpaceBoppSuperoperator(_BoppActor, HilbertSpaceObject, NotAnOperator):
    """
    The Hilbert-space Bopp superoperator (HSBS). See Appendix B in https://arxiv.org/abs/2509.17106.
    """
    Hilbert = True
HSBS = HilbertSpaceBoppSuperoperator
    
class PhaseSpaceBoppOperator(_BoppActor, PhaseSpaceObject, NotAScalar):
    """
    The phase-space Bopp operator (PSBO). See Appendix B in https://arxiv.org/abs/2509.17106.
    """
    Hilbert = False
    
PSBO = PhaseSpaceBoppOperator