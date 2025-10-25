from ._internal import constants as m

s = m.CahillGlauberSParameter()
hbar = m.ReducedPlanckConstant()
zeta = m.AlphaScalingParameter()
pi = m.piTranscendentalNumber()

del m

###
# PATCHES 

from ._internal.patches import apply_patches
apply_patches()
del apply_patches

###

from ._internal.multiprocessing import MP_CONFIG

###

from .objects.scalars import q, p, alpha, alphaD, W
from .objects.operators import (qOp, pOp, createOp, annihilateOp, rho)

from .bopp import HilbertSpaceBoppSuperoperator, PhaseSpaceBoppOperator
from .cg import CGTransform, iCGTransform, CGProps
from .eom import LindbladMasterEquation
from .manipulations import (
    alpha2qp, qp2alpha, sc2op, op2sc, dagger, explicit, express,
    normal_ordered_equivalent
)
from .ordering import sOrdering, normal_order, weyl_order, antinormal_order
from .quantization import naive_quantize, s_quantize
from .star import Star, HattedStar
from .utils import (
    collect_by_derivative, derivative_not_in_num, get_N, get_random_poly,
    opder2comm
)