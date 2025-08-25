import sympy as sp

from ._internal.basic_routines import operation_routine
from ._internal.cache import sub_cache
from ._internal.multiprocessing import _mp_helper

from .objects.base import Base
from .objects import scalars
from .objects.scalars import q, p, alpha, alphaD, W
from .objects.operators import qOp, pOp, annihilateOp, createOp, Operator, rho

###

def define(expr : sp.Expr) -> sp.Expr:
    """
    Given a composite expression `expr`, call the `.define` method
    where applicable.
    """
    expr = sp.sympify(expr)
    expr_defined = expr.subs({A: A.define() for A in expr.atoms(Base)})
    return sp.expand(expr_defined)

def qp2a(expr : sp.Expr) -> sp.Expr:
    def get_subs_expr(A : scalars.Scalar | Operator):
        if isinstance(A, scalars.Scalar):
            a, ad = scalars.alpha(A.sub), scalars.alphaD(A.sub)
        else:
            a, ad = annihilateOp(A.sub), createOp(A.sub)
            
        mu = scalars.mu
        mu_conj = sp.conjugate(mu)
        hbar = scalars.hbar
        
        if isinstance(A, (scalars.q, qOp)):
            out = mu*a + mu_conj*ad
        else:
            out = sp.I*mu*mu_conj*(mu*ad - mu_conj*a)
            
        out *= sp.sqrt(2*hbar) / (mu**2 + mu_conj**2)
        
        return out
        
    sub_dict = {}
    for sub in sub_cache:
        sub_dict[scalars.q(sub)] = get_subs_expr(scalars.q(sub))
        sub_dict[scalars.p(sub)] = get_subs_expr(scalars.p(sub))
        sub_dict[qOp(sub)] = get_subs_expr(qOp(sub))
        sub_dict[pOp(sub)] = get_subs_expr(pOp(sub))
        
    return sp.expand(expr.subs(sub_dict))

def dagger(expr : sp.Expr) -> sp.Expr:
    
    def treat_add(A : sp.Expr):
        return sp.Add(*_mp_helper(A.args, dagger))
    
    def treat_pow(A : sp.Expr):
        return dagger(A.args[0]) ** A.args[1]
    
    def treat_mul(A : sp.Expr):
        return sp.Mul(*list(reversed(_mp_helper(A.args, dagger))))
    
    return operation_routine(expr,
                            "Dagger",
                            [],
                            [],
                            {Operator : lambda A: sp.conjugate(A)},
                            {Operator : lambda A: A.dagger(),
                                sp.Add : treat_add,
                                sp.Pow : treat_pow,
                                sp.Mul : treat_mul}
                            )
    
def explicit(expr: sp.Expr) -> sp.Expr:
    from .ordering import sOrdering
    return expr.replace(lambda A: isinstance(A, sOrdering),
                        lambda A: A.explicit())
    
def express(expr : sp.Expr, t=1, explicit=True) -> sp.Expr:
    from .ordering import sOrdering
    return expr.replace(lambda A: isinstance(A, sOrdering),
                        lambda A: A.express(t=t, explicit=explicit))

# =============================
# Operator-Scalar substitutions
# =============================

def get_op2sc_dict() -> dict:
    return {op(sub) : sc(sub) 
            for sub in sub_cache 
            for op,sc in [[qOp, q],
            [pOp, p],
            [annihilateOp, alpha], 
            [createOp, alphaD]]} | {rho : W}

def get_sc2op_dict() -> dict:
    return {v:k for k,v in get_op2sc_dict().items()}

def op2sc(expr : sp.Expr) -> sp.Expr:
    return expr.subs(get_op2sc_dict())

def sc2op(expr : sp.Expr) -> sp.Expr:
    return expr.subs(get_sc2op_dict())