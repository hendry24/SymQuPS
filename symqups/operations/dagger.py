import sympy as sp

from ..objects.operators import Operator
from ..utils._internal._basic_routines import _operation_routine
from ..utils.multiprocessing import _mp_helper

def dagger(expr : sp.Expr | Operator):
    
    def treat_add(A : sp.Expr):
        return sp.Add(*_mp_helper(A.args, dagger))
    
    def treat_pow(A : sp.Expr):
        return dagger(A.args[0]) ** A.args[1]
    
    def treat_mul(A : sp.Expr):
        return sp.Mul(*list(reversed(_mp_helper(A.args, dagger))))
    
    return _operation_routine(expr,
                                "Dagger",
                                (),
                                {Operator : lambda A: sp.conjugate(A)},
                                {Operator : lambda A: A.dagger(),
                                 sp.Add : treat_add,
                                 sp.Pow : treat_pow,
                                 sp.Mul : treat_mul}
                                )