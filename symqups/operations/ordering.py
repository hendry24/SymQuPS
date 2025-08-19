import sympy as sp

from .. import s
from ..objects.base import qpTypePSO, alphaTypePSO, PhaseSpaceObject
from ..objects.scalars import alpha, alphaD
from ..objects.cache import _sub_cache
from ..objects.operators import Operator, annihilateOp, createOp
from ..utils._internal import _operation_routine, _invalid_input
from ..utils.multiprocessing import _mp_helper
from ..utils.algebra import qp2a

def s_ordered(sub, m, n):
    """
    Compute the s-ordering of the quantization of
    `alphaD(sub)**m * alpha(sub)**n`. Since we 
    can do whatever ordering when we quantize the
    expression, we choose normal ordering as the
    initial operator ordering, i.e. we initially
    have `createOp(sub)**m * annihilateOp(sub)**n`.
    Eq. (5.12) of https://doi.org/10.1103/PhysRev.177.1857
    can then be used to obtain its s-ordering.
    """
    
    if (m==0) and (n==0):
        return 1
    
    out = 0
    for k in range(min(m, n) + 1):
        out += (sp.factorial(k)
                * sp.binomial(n, k)
                * sp.binomial(m, k)
                * ((1-s.val)/2)**k
                * createOp(sub)**(m-k) * annihilateOp(sub)**(n-k)
        )
    return out