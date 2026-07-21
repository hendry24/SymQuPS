import sympy as sp

from .cg import iCGTransform

def s_quantize(expr : sp.Expr) -> sp.Expr:
    """
    Canonical-quantize ``expr`` with s-ordering. Equivalent to ``iCGTransform(expr)``.
    """
    return iCGTransform(expr)

def _set_s_and_quantize(expr, s):
    from . import s as CahillGlauberS
    original_s = CahillGlauberS.val
    
    CahillGlauberS.val  = s
    
    out = iCGTransform(expr)
    
    CahillGlauberS.val = original_s
    
    return out

def normal_quantize(expr : sp.Expr) -> sp.Expr:
    """
    Canonical-quantize ``expr`` with normal ordering.
    """
    return _set_s_and_quantize(expr, 1)

def Weyl_quantize(expr : sp.Expr) -> sp.Expr:
    """
    Canonical-quantize ``expr`` with Weyl or symmetric ordering.
    """
    return _set_s_and_quantize(expr, 0)

def antinormal_quantize(expr : sp.Expr) -> sp.Expr:
    """
    Canonical-quantize ``expr`` with antinormal ordering.
    """
    return _set_s_and_quantize(expr, -1)