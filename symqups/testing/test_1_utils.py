import pytest
import sympy as sp
import random

from symqups.objects.scalars import Scalar, q, p, alpha, alphaD, W, hbar, mu
from symqups.objects.operators import Operator, qOp, pOp, createOp, annihilateOp
from symqups.utils.multiprocessing import _mp_helper, MP_CONFIG
from symqups.utils.algebra import get_random_poly, define, qp2a

def mp_helper_foo(x):
        return x+2

@pytest.mark.fast
@pytest.mark.order(2)
def test_mp_helper():
    inpt = [1, sp.Symbol("x"), Scalar(),
            Operator(), W()]
    
    global MP_CONFIG
    enable_default = MP_CONFIG["enable"]
    MP_CONFIG["min_num_args"] = 0

    for enable in [True, False]:
        MP_CONFIG["enable"] = enable
        assert (_mp_helper(inpt, mp_helper_foo) 
                == list(map(mp_helper_foo, inpt)))
    
    MP_CONFIG["enable"] = enable_default

@pytest.mark.full
@pytest.mark.order(3)
class TestAlgebra():
    def test_define_qp2a(self):
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
            assert expr_def.has(qq, pp)
            assert not(expr_def).has(a, ad)
            assert not(expr_qp2a).has(qq, pp)
            assert expr_qp2a.has(a, ad)
            assert sp.simplify(define(expr_qp2a) -  expr_def) == 0
            assert sp.simplify(qp2a(expr_def) - expr_qp2a) == 0