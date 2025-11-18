import sympy as sp

from . import s as CahillGlauberS
from .cg import iCGTransform

def s_quantize(expr : sp.Expr) -> sp.Expr:
    return iCGTransform(expr)

def _set_s_and_quantize(expr, s):
    original_s = CahillGlauberS.val
    CahillGlauberS.val  = s
    out = iCGTransform(expr)
    CahillGlauberS.val = original_s
    return out            

def normal_quantize(expr : sp.Expr) -> sp.Expr:
    return _set_s_and_quantize(expr, 1)

def Weyl_quantize(expr : sp.Expr) -> sp.Expr:
    return _set_s_and_quantize(expr, 0)

def antinormal_quantize(expr : sp.Expr) -> sp.Expr:
    return _set_s_and_quantize(expr, -1)