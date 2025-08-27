from ._internal import constants as m

s = m.CahillGlauberSParameter()
hbar = m.ReducedPlanckConstant()
zeta = m.AlphaScalingParameter()
pi = m.piTranscendentalNumber()

del m

from .utils import enable_Mul_patch
enable_Mul_patch()

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