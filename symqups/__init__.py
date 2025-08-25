from ._internal.s_parameter import CahillGlauberSParameter

s = CahillGlauberSParameter()

del CahillGlauberSParameter

###

def enable_Mul_patch():
    import sympy as sp
    from ._internal.mul import patched_Mul_flatten

    sp.Mul.flatten = patched_Mul_flatten

def disable_Mul_patch():
    import sympy as sp
    from ._internal.mul import original_Mul_flatten

    sp.Mul.flatten = original_Mul_flatten

enable_Mul_patch()

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