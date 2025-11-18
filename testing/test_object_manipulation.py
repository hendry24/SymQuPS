import pytest
import dill
import random
from copy import deepcopy
from utils import arithmetic_test

import sympy as sp
sMul = deepcopy(sp.Mul)

from symqups._internal.grouping import alphaType, qpType
from symqups.objects.scalars import (Scalar, q, p, t, W, alpha, alphaD)
from symqups.objects.operators import (Operator, qOp, pOp, createOp, annihilateOp,
                                        densityOp, rho)
from symqups.utils import get_random_poly
from symqups.ordering import sOrdering, normal_order

from symqups import hbar, zeta

hbar = hbar.val
zeta = zeta.val

# TESTED FUNCTIONALITIES
########################

from symqups.manipulations import (
    dagger, qp2alpha, alpha2qp, normal_ordered_equivalent, 
    explicit_sOrdering, express_sOrdering, s_ordered_equivalent,
    Derivative, Commutator, op2sc, sc2op
)

###
    
@pytest.mark.fast
def test_core_arithmetic():
    for A in [Scalar(), Operator()]:
        arithmetic_test(A)

@pytest.mark.fast
def test_compound_expressions_with_objects():
    for A in [Scalar(), Operator()]:
        assert dill.loads(dill.dumps(sp.Expr(A))) == sp.Expr(A)
        assert sp.Expr(A).args[0] == A
        assert dill.loads(dill.dumps(sp.Function("F")(A))) == sp.Function("F")(A)
        assert A in sp.Function("F")(A).free_symbols
    
    F = sp.Function("F")(Scalar(), Operator())
    assert dill.loads(dill.dumps(F)) == F

@pytest.mark.full
def test_alpha2qp_and_qp2alpha():
    
    assert sp.conjugate(alpha()) == alphaD()
    assert sp.conjugate(alphaD()) == alpha()

    sub = random.randint(0, 100)
       
    assert qp2alpha(alpha(sub)) == alpha(sub)
    assert qp2alpha(annihilateOp(sub)) == annihilateOp(sub)
    
    zetaD = sp.conjugate(zeta)
    
    for qq, pp, a, ad in zip([q(sub), qOp(sub)], 
                                [p(sub), pOp(sub)], 
                                [alpha(sub), annihilateOp(sub)], 
                                [alphaD(sub), createOp(sub)]):
        assert sp.expand(qp2alpha(qq) - sp.sqrt(2*hbar)*(zeta*a + zetaD*ad)/(zeta**2+zetaD**2)) == 0
        assert sp.expand(qp2alpha(pp) - sp.sqrt(2*hbar)*sp.I*zeta*zetaD*(zeta*ad-zetaD*a)/(zeta**2+zetaD**2)) == 0
    
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
        expr = get_random_poly(obj_lst)
        expr_def = alpha2qp(expr)
        expr_qp2a = qp2alpha(expr)
        assert not(expr_def.has(alphaType)) and expr_def.has(qpType)
        assert not(expr_qp2a.has(qpType)) and expr_qp2a.has(alphaType)
        assert sp.simplify(alpha2qp(expr_qp2a) -  expr_def) == 0
        assert sp.simplify(qp2alpha(expr_def) - expr_qp2a) == 0

@pytest.mark.fast
def test_op2sc_and_sc2op():
    aOp = annihilateOp()
    adOp = createOp()
    a = alpha()
    ad = alphaD()
    
    assert all([
        op2sc(1) == 1,
        op2sc(a) == a,
        op2sc(aOp) == a,
        op2sc(a*aOp) == a**2,
        op2sc(aOp+adOp) == a+ad,
        sc2op(1) == 1,
        sc2op(aOp) == aOp,
        sc2op(a) == aOp,
        sc2op(a*aOp) == aOp**2,
        sc2op(a+ad) == aOp+adOp
    ])
    
    rand_poly = get_random_poly([aOp, adOp])
    assert (sc2op(op2sc(rand_poly)) - normal_order(rand_poly)).expand() == 0

@pytest.mark.full
def test_dagger():
    assert dagger(1) == 1
    assert dagger(1+1j) == sp.sympify(1-1j)
    assert dagger(alpha()) == alphaD()
    assert dagger(alphaD()) == alpha()
    
    for herm_op in [qOp(), pOp(), densityOp()]:
        assert dagger(herm_op) == herm_op

    assert dagger(annihilateOp()) == createOp()
    assert dagger(createOp()) == annihilateOp()   

    rand_poly = get_random_poly(objects = (1, sp.Symbol("x"), qOp(), annihilateOp(),
                                            createOp(), annihilateOp(),
                                            alpha(), alphaD()),
                                coeffs = list(range(10)) + sp.symbols([]),
                                n_terms=10)
    assert (dagger(dagger(rand_poly)) - rand_poly).expand() == 0
    
@pytest.mark.full
def test_normal_ordered_equivalent():
    aop = [annihilateOp(i) for i in range(3)]
    adop = [createOp(i) for i in range(3)]
    x = sp.Symbol("x")
    
    assert normal_ordered_equivalent(1) == 1
    assert normal_ordered_equivalent(x) == x
    assert normal_ordered_equivalent(adop[0]) == adop[0]
    assert normal_ordered_equivalent(rho) == rho
    
    assert normal_ordered_equivalent(adop[0]*aop[0]) == adop[0]*aop[0]
    assert normal_ordered_equivalent(aop[0]*adop[0]) == 1 + adop[0]*aop[0]
    assert normal_ordered_equivalent(aop[0]*adop[0]-1) == adop[0]*aop[0]
    
    assert normal_ordered_equivalent(adop[0]**2) == adop[0]**2
    assert normal_ordered_equivalent(aop[0]*adop[0]**2) == 2*adop[0]+adop[0]**2*aop[0]
    
    assert (sp.simplify(sp.expand(normal_ordered_equivalent(aop[0]*adop[0]*aop[1]*adop[1])) 
             - sp.expand((1+adop[0]*aop[0])*(1+adop[1]*aop[1])))
            == 0)
    
@pytest.mark.full
def test_explicit_and_express_sOrdering():
    a = annihilateOp()
    ad = createOp()

    expr = 2*sOrdering(a*ad, s=1)+3
    assert not(explicit_sOrdering(expr).has(sOrdering))
    
    expr = 2*sOrdering(a*ad, s=0.5)+3
    assert explicit_sOrdering(expr).has(sOrdering)
    
    expr = 2*sOrdering(sp.exp(a*ad), s=1)**2+3
    assert explicit_sOrdering(expr) == 2*sOrdering(sp.exp(a*ad), s=1)**2+3

    ###
    
    expr = 2*sOrdering(a*ad, s = 1)
    assert express_sOrdering(expr, t = 0.2).has(sOrdering)

    expr = 2*sOrdering(a*ad, s = 1)
    assert not(express_sOrdering(expr, t = -1).has(sOrdering))
    assert express_sOrdering(expr, t = -1, explicit=False).has(sOrdering)
    
@pytest.mark.fast
def test_s_ordered_equivalent():
    a = annihilateOp()
    ad = createOp()
    
    a_lst = [annihilateOp(i) for i in range(4)]
    ad_lst = [createOp(i) for i in range(4)]
    
    assert s_ordered_equivalent(1) == 1
    assert s_ordered_equivalent(a) == a
    assert s_ordered_equivalent(ad) == ad
    
    from symqups import s
    assert s_ordered_equivalent(a*ad) == (sOrdering(a*ad) + (s.val+1)/2).expand()
    
    rand_poly = get_random_poly(a_lst+ad_lst)
    assert (express_sOrdering(s_ordered_equivalent(rand_poly),1,True)
            - normal_ordered_equivalent(rand_poly)).expand() == 0
    
@pytest.mark.fast
def test_derivative():
    aOp, adOp = annihilateOp(), createOp()
    a, ad = alpha(), alphaD()
    x = sp.Symbol("x")
    f = sp.Function("f")(a,ad)
    F = sp.Function("F")(aOp, adOp)
    X = sp.Function("X")(a, ad, aOp, adOp, x)
    tt = t()
    
    assert isinstance(Derivative(f, a), sp.Derivative)
    
    assert Derivative(1).doit() == 1
    assert Derivative(1, a).doit() == 0
    assert Derivative(f, a).doit() == sp.Derivative(f,a)
    assert Derivative(F, a).doit() == 0
    assert Derivative(F, adOp) == Commutator(aOp, F)
    assert Derivative(f, adOp) == Commutator(aOp, f)
    assert Derivative(X, x).doit() == sp.Derivative(X, x)
    assert Derivative(X, adOp, x) == Derivative(Commutator(aOp, X), x)
    
    assert Derivative(rho, tt).doit() != 0
    assert Derivative(rho, tt, adOp, x) == Derivative(Commutator(aOp, rho), 
                                                      t(), x)