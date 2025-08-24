import pytest
import dill
import random
from copy import deepcopy

import sympy as sp
sMul = deepcopy(sp.Mul)

from symqups.objects.base import alphaTypePSO, qpTypePSO
from symqups.objects.scalars import (hbar, mu, Scalar, q, p, t, W, alpha, alphaD,
                                    _Primed, _deprime, _DerivativeSymbol)
from symqups.objects.operators import (Operator, qOp, pOp, createOp, annihilateOp,
                                        densityOp, rho)
from symqups.operations.dagger import dagger
from symqups.utils.algebra import get_random_poly, define, qp2a


def arithmetic(A):
    assert A+2 == 2+A
    assert 2+A == A+2
    assert A-2 == sp.expand(-(2-A))
    assert A*2 == 2*A
    assert A/2 == (1/(2/A)).doit()
    assert sp.expand(A**(-2)) == 1/(A**2)
    assert A**(0) == 1
    assert sp.sqrt(A)
    
@pytest.mark.fast
def test_core_arithmetic():
    for A in [Scalar(), Operator(), 
              _Primed(alpha()), _DerivativeSymbol(_Primed(alpha()))]:
        arithmetic(A)

@pytest.mark.fast
def test_compound_expressions_with_objects():
    for A in [Scalar(), Operator(), 
              _Primed(alpha()), _DerivativeSymbol(_Primed(alpha()))]:
        assert dill.loads(dill.dumps(sp.Expr(A))) == sp.Expr(A)
        assert sp.Expr(A).args[0] == A
        assert dill.loads(dill.dumps(sp.Function("F")(A))) == sp.Function("F")(A)
        assert A in sp.Function("F")(A).free_symbols

@pytest.mark.fast
def test_multiplication_reordering():
    a_op_1 = annihilateOp(1)
    ad_op_1 = createOp(1)
    a_op_2 = annihilateOp(2)
    ad_op_2 = createOp(2)
    
    # Same 'sub' -> do not commute
    assert a_op_1*ad_op_1 != ad_op_1*a_op_1
    assert ad_op_2*a_op_2 != a_op_2*ad_op_2
    
    # Different 'sub's > commute
    assert a_op_1*ad_op_2 == ad_op_2*a_op_1
    assert ad_op_1*a_op_2 == a_op_2*ad_op_1
    
    # Not the case for other noncommutative objects. 
    a_p_1 = _Primed(alpha(1))
    a_p_2 = _Primed(alpha(2))
    da_1 = _DerivativeSymbol(a_p_1)
    da_2 = _DerivativeSymbol(a_p_2)
    
    assert a_p_1*da_1 != da_1*a_p_1
    assert a_p_2*da_2 != da_2*a_p_2
    
    assert a_p_1*da_2 != da_2*a_p_1
    assert a_p_2*da_1 != da_1*a_p_2
    
@pytest.mark.fast
def test_deprime():
    rand_poly = get_random_poly([q(), p(), alpha(), alphaD(), Scalar()],
                                [1, sp.Symbol("x"), sp.Symbol("y"), sp.exp(sp.Symbol("z"))],
                                dice_throw = 3)
    assert sp.expand(_deprime(_Primed(rand_poly)) - rand_poly) == 0

@pytest.mark.full
def test_define_and_qp2a():
    for obj, expanded in zip([alpha(), alphaD()],
                                [(q()*mu + sp.I*p()/mu) / sp.sqrt(2*hbar),
                                (q()*sp.conjugate(mu) - sp.I*p()/sp.conjugate(mu)) / sp.sqrt(2*hbar)]):
        assert sp.expand(obj.define() - expanded) == 0
    
    assert sp.conjugate(alpha()) == alphaD()
    assert (sp.conjugate(alpha().define()) - alphaD().define()).expand() == 0
    assert sp.conjugate(alphaD()) == alpha()
    assert (sp.conjugate(alphaD().define()) - alpha().define()).expand() == 0

    sub = random.randint(0, 100)
    
    assert sp.expand(define(alpha(sub)) - alpha(sub).define()) == 0
    assert sp.expand(define(annihilateOp(sub)) - annihilateOp(sub).define()) == 0
    
    assert qp2a(alpha(sub)) == alpha(sub)
    assert qp2a(annihilateOp(sub)) == annihilateOp(sub)
    
    muD = sp.conjugate(mu)
    
    for qq, pp, a, ad in zip([q(sub), qOp(sub)], 
                                [p(sub), pOp(sub)], 
                                [alpha(sub), annihilateOp(sub)], 
                                [alphaD(sub), createOp(sub)]):
        assert sp.expand(qp2a(qq) - sp.sqrt(2*hbar)*(mu*a + muD*ad)/(mu**2+muD**2)) == 0
        assert sp.expand(qp2a(pp) - sp.sqrt(2*hbar)*sp.I*mu*muD*(mu*ad-muD*a)/(mu**2+muD**2)) == 0
    
    qp_sc_lst = []
    qp_op_lst = []
    a_sc_lst = []
    a_op_lst = []
    for sub in range(5):
        qp_sc_lst.extend([q(sub), p(sub)])
        qp_op_lst.extend([qOp(sub), pOp(sub)])
        a_sc_lst.extend([alpha(sub), alphaD(sub)])
        a_op_lst.extend([annihilateOp(sub), createOp(sub)])
    
    for obj_lst, (qq, pp, a, ad) in zip([qp_sc_lst+a_sc_lst, qp_op_lst+a_op_lst],
                                            [(q, p, alpha, alphaD), 
                                            (qOp, pOp, annihilateOp, createOp)]):
        expr = get_random_poly(obj_lst, dice_throw=3)
        expr_def = define(expr)
        expr_qp2a = qp2a(expr)
        assert not(expr_def.has(alphaTypePSO)) and expr_def.has(qpTypePSO)
        assert not(expr_qp2a.has(qpTypePSO)) and expr_qp2a.has(alphaTypePSO)
        assert sp.simplify(define(expr_qp2a) -  expr_def) == 0
        assert sp.simplify(qp2a(expr_def) - expr_qp2a) == 0

@pytest.mark.full
def test_dagger():
    assert dagger(1) == 1
    
    for herm_op in [qOp(), pOp(), densityOp()]:
        assert dagger(herm_op) == herm_op

    assert dagger(annihilateOp()) == createOp()
    assert (dagger(annihilateOp().define())-createOp().define()).expand() == 0
    assert dagger(createOp()) == annihilateOp()            
    assert (dagger(createOp().define())-annihilateOp().define()).expand() == 0

    rand_poly = get_random_poly(objects = (1, sp.Symbol("x"), qOp(), annihilateOp(),
                                            createOp(), annihilateOp()),
                                coeffs = list(range(10)) + sp.symbols([]),
                                dice_throw = 3)
    assert (dagger(dagger(rand_poly)) - rand_poly).expand() == 0