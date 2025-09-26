import sympy as sp
from typing import Tuple

from .cache import sub_cache
from .grouping import PhaseSpaceVariable

def collect_psv_in_monomial_by_sub(expr: sp.Expr) -> Tuple[sp.Expr, dict]:
    others = sp.Number(1)
    sub_factor_dict = {sub : sp.Integer(1) for sub in sub_cache}
    
    args = expr.args if expr.is_Mul else [expr]
    for arg in args:
        psv_atom = arg.atoms(PhaseSpaceVariable)
        if psv_atom:
            sub = list(psv_atom)[0]
            sub_factor_dict[sub] *= arg
        else:
            others *= arg
    
    return others, sub_factor_dict
    