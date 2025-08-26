import sympy as sp

from ._internal.basic_routines import operation_routine
from ._internal.operator_handling import is_universal, separate_term_oper_by_sub, get_oper_sub
from ._internal.cache import sub_cache
from ._internal.multiprocessing import mp_helper

from .objects.base import Base
from .objects import scalars
from .objects.scalars import q, p, alpha, alphaD, W, _Primed
from .objects.operators import qOp, pOp, annihilateOp, createOp, Operator, rho

###

def _deprime(expr : sp.Expr):
    subs_dict = {X : X.base for X in expr.atoms(_Primed)}
    return expr.subs(subs_dict)

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
        return sp.Add(*mp_helper(A.args, dagger))
    
    def treat_pow(A : sp.Expr):
        return dagger(A.args[0]) ** A.args[1]
    
    def treat_mul(A : sp.Expr):
        return sp.Mul(*list(reversed(mp_helper(A.args, dagger))))
    
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


# Operator-Scalar substitutions
# =============================

def _get_op2sc_dict() -> dict:
    return {op(sub) : sc(sub) 
            for sub in sub_cache 
            for op,sc in [[qOp, q],
            [pOp, p],
            [annihilateOp, alpha], 
            [createOp, alphaD]]} | {rho : W}

def _get_sc2op_dict() -> dict:
    return {v:k for k,v in _get_op2sc_dict().items()}

def op2sc(expr : sp.Expr) -> sp.Expr:
    return expr.subs(_get_op2sc_dict())

def sc2op(expr : sp.Expr) -> sp.Expr:
    return expr.subs(_get_sc2op_dict())

# Normal-ordered equivalent
###########################

def normal_ordered_equivalent(expr : sp.Expr) -> sp.Expr:
    """
    Returns the normal-ordered **equivalent** of the input, provided
    that it is a polynomial in 'Operator' objects. If a term is not
    a polynomial in 'Operator', it is not treated by this function.
    
    This function implements Blasiak's explicit formulae to obtain
    the normal-ordered equivalent of a monomial in the creation/annihilation
    operators, similar to [`pybolano`](https://github.com/hendry24/pybolano).
    """
    
    def treat_add(A : sp.Expr) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, normal_ordered_equivalent))
    
    def treat_mul(A : sp.Expr) -> sp.Expr:
        if not(expr.is_polynomial(Operator)) or is_universal(expr):
            return A
        
        if len(get_oper_sub(A)) == 1:
            return treat_mul_single_sub(A)
        
        return sp.Mul(*mp_helper(separate_term_oper_by_sub(A),
                                 normal_ordered_equivalent))
        
    def treat_mul_single_sub(A : sp.Expr) -> sp.Expr:
        if not(A.has(Operator)) or isinstance(A, (Operator, sp.Pow)):
            return A
                
        r = [] if A.args[0].has(createOp) else [sp.Number(0)]
        s = []
        for arg in A.args:
            if isinstance(arg, Operator):
                exponent = 1
            else:
                exponent = arg.args[1]
            
            if arg.has(createOp):
                r.append(exponent)
            else:
                s.append(exponent)
    
        if len(r) != len(s):
            s.append(sp.Number(0))
    
        r.append(sp.Number(0))
        s.append(sp.Number(0))
    
        r = list(reversed(r))
        s = list(reversed(s))
        
        # Excess
        d = []
        sum_val = sp.Number(0)
        for r_m, s_m in zip(r, s):
            sum_val += (r_m - s_m)
            d.append(sum_val)
        
        def S_rs(s, d, k):
            # Generalized Stirling number.
            sum_val = sp.Number(0)
            for j in range(k+1):
                prod_val = sp.Number(1)
                for m in range(1, len(s)):
                    prod_val *= sp.FallingFactorial(d[m-1]+j, 
                                                    s[m])
                sum_val += sp.binomial(k,j) * sp.Number(-1)**(k-j) * prod_val
            return sum_val / sp.factorial(k)
        
        a = list(A.atoms(annihilateOp))[0]
        ad = list(A.atoms(createOp))[0]
        
        if d[-1] >= 0:
            R, S, D = r, s, d
            k_lst = range(s[1], sum(s)+1)
        else:
            k_lst = range(r[-1], sum(r)+1)
            
            R = [sp.Number(0)] + list(reversed(s[1:]))
            S = [sp.Number(0)] + list(reversed(r[1:]))
            D = []
            sum_val = sp.Number(0)
            for R_m, S_m in zip(R, S):
                sum_val += (R_m-S_m)
                D.append(sum_val)
        
        out = 0
        for k in k_lst:
            out += S_rs(S, D, k) * ad**k * a**k
            
        if d[-1] >= 0:
            out = ad**d[-1] * out
        else:
            out = out * a**(-d[-1])
        
        return sp.expand(out) 
        
    expr = qp2a(sp.sympify(expr))
    return operation_routine(expr,
                             "normal_ordered_equivalent",
                             [],
                             [],
                             {Operator : expr},
                             {sp.Add : treat_add,
                              sp.Mul : treat_mul,
                              (Operator, sp.Pow, sp.Function) : expr})