import sympy as sp

class ObjectGroup(sp.Basic):
    """
    Base class for the package's grouping.
    """
    def __new__(cls, *args, **kwargs):
        if cls in [ObjectGroup, qpType, alphaType,
                   PhaseSpaceObject, PrimedPSO, UnBoppable,
                   HilbertSpaceObject]:
            raise TypeError("For grouping only. Instantiation is prohibited.")
        return super().__new__(cls, *args, **kwargs)
    
class qpType(ObjectGroup): # complicance with sympy functionalities.
    """
    This object is related to the canonical phase-space variables `(q,p)`.
    """
    pass

class alphaType(ObjectGroup):
    """
    This object is related to the complex phase-space amplitude `alpha`.
    """
    pass

###

class PhaseSpaceObject(ObjectGroup):
    """
    This object lives in the quantum phase space.
    """
    pass

class PhaseSpaceVariable(PhaseSpaceObject):
    """
    This object is a phase space variable.
    """
    pass

###

class PrimedPSO(ObjectGroup):
    """
    This object is the primed version of the `PhaseSpaceObject`. Not
    physical, but `sympy`-convenient.
    """
    pass

###

class UnBoppable(ObjectGroup):
    """
    This object cannot be Bopp-shifted.
    """
    pass

class HilbertSpaceObject(UnBoppable):
    """
    This object lives in the Hilbert Space.
    """
    pass

###

class NotAnOperator(ObjectGroup):
    """
    This object contains 'Operator', but is not one. 
    """
    pass

class NotAScalar(ObjectGroup):
    """
    This object contains 'Scalar', but is not one. 
    """
    pass

###
    
class Defined(sp.Basic):
    """
    This object has the `definition` class property.
    """
    @staticmethod
    def _definition():
        return NotImplementedError
    definition = _definition()
    
###

class _AddOnlyExpr(sp.Expr):
    def __pow__(self, other):
        raise NotImplementedError()
    __rpow__ = __pow__
    __mul__ = __pow__
    __rmul__ = __pow__
    __sub__ = __pow__
    __rsub__ = __pow__
    __truediv__ = __pow__
    __rtruediv__ = __pow__
    
class _ReadOnlyExpr(_AddOnlyExpr):
    def __add__(self, other):
        raise NotImplementedError()
    __radd__ = __add__