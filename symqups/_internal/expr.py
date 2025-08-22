import sympy as sp

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