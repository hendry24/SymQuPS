from ._internal import constants as m

s = m.CahillGlauberSParameter()
hbar = m.ReducedPlanckConstant()
zeta = m.AlphaScalingParameter()
pi = m.piTranscendentalNumber()

del m

###
# PATCHES 

import sympy as sp
from ._internal.patches import patched_Mul_flatten, PatchedDerivative

sp.Mul.flatten = patched_Mul_flatten
sp.Derivative = PatchedDerivative

del sp, patched_Mul_flatten, PatchedDerivative

###

from ._internal.multiprocessing import MP_CONFIG

###

# from .objects.scalars import q, p, alpha, alphaD, P
# from .objects.hilbert_operators import (qOp, pOp, 
#                                createOp, annihilateOp, 
#                                Dagger, rho)

# from .transforms.star_product import Bopp, Star

# from .transforms.wigner_transform import WignerTransform
# from .keep.eom import LindbladMasterEquation

# from .utils.multiprocessing import MP_CONFIG
# from .utils.grouping import collect_by_derivative, derivative_not_in_num

# from .objects.scalars import *