import sympy as sp

###

from . import s

###

from . import disable_Mul_patch
disable_Mul_patch()
del disable_Mul_patch

###

from .objects.scalars import q, p, alpha, alphaD, W
q = q()
p = p()
a = alpha()
ad = alphaD()

del alpha, alphaD

###

from .objects.operators import qOp, pOp, annihilateOp, createOp, rho
qOp = qOp()
pOp = pOp()
aOp = annihilateOp()
adOp = createOp()

del annihilateOp, createOp

###

from .quantization import naive_quantize, s_quantize
from .star_product import Bopp, Star
from ._internal.multiprocessing import MP_CONFIG
from .manipulations import alpha2qp, qp2alpha
from .utils import collect_by_derivative