import sympy as sp

from ._internal.multiprocessing import mp_helper
from ._internal.basic_routines import operation_routine
from ._internal.grouping import (PhaseSpaceVariable, PhaseSpaceObject, Defined, 
                                 HilbertSpaceObject, NotAnOperator, NotAScalar,
                                 PhaseSpaceVariableOperator)
from ._internal.cache import sub_cache

from .objects.scalars import W, StateFunction, alpha, alphaD
from .objects.operators import Operator, densityOp, rho

from .star import Star, HattedStar
from .ordering import sOrdering
from .manipulations import qp2alpha, op2sc, alpha2qp, sc2op, Commutator, express
from .utils import get_N, _treat_der_template

from . import s as CahillGlauberS
from . import pi

###

_kernel_string = r"\mathcal{T}_s\left(\bm{\alpha}\right) = "
_kernel_string += r"\int_{\mathbb{R}^{2N}}\mathrm{d}\bm{\beta}"
_kernel_string += r"\exp\left(\frac{s}{2}\left|\bm{\beta}\right|^2\right)"
_kernel_string += r"\exp\left(\bm{\beta}\cdot\bm{\hat{a}}^\dagger -\overline{\bm{\beta}}\cdot\hat{\bm{a}}\right)"
_kernel_string +=  r"\exp\left(\bm{\alpha}\cdot\overline{\bm{\beta}} - \overline{\bm{\alpha}}\cdot\bm{\beta} \right)"

def _CGT_str(x : str, var : bool = False) -> str:
    out = r"\mathcal{W}_s\left[%s\right]" % x
    if var:
        out += r"\left(\bm{\alpha}, \bm{\alpha}^*\right)"
    return out

def _iCGT_str(x : str, var : bool = False) -> str:
    out = r"\mathcal{W}^{-1}_s\left[%s\right]" % x
    if var:
        out += r"\left(\hat{\bm{a}}, \hat{\bm{a}}^\dagger\right)"
    return out

###

class CGTransform(sp.Expr, PhaseSpaceObject, Defined, NotAnOperator):
    
    @staticmethod
    def _definition():
        lhs = sp.Symbol(_CGT_str(r"\hat{A}"))
        rhs = sp.Symbol(r"\mathrm{tr}\left(\hat{A}\mathcal{T}_{-s}\left(\bm{\alpha}\right)\right), \quad %s" % _kernel_string)
        return sp.Equality(lhs, rhs)
    definition = _definition()
    
    def __new__(cls, expr : sp.Expr, *_vars):
        """
        oper -> quantum ps vars
        """
        if expr.is_Equality:
            from .eom import LindbladMasterEquation
            lhs = expr.lhs
            rhs = expr.rhs
            if isinstance(expr, LindbladMasterEquation):
                lhs /= pi.val
                rhs /= pi.val
            return sp.Equality(CGTransform(lhs),
                               CGTransform(rhs)).doit().expand()
        
        if not(_vars):
            _vars = sub_cache._get_alphaType_scalar()
        
        def treat_add(A : sp.Expr) -> sp.Expr:
            return sp.Add(*mp_helper(A.args, CGTransform))
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
            return Star(*mp_helper(A.args, CGTransform))
            
        def treat_sOrdering(A : sOrdering) -> sp.Expr:
            if (A.args[1] != CahillGlauberS.val):
                if not(A.args[0].is_polynomial(PhaseSpaceVariableOperator)):
                    return make(A)
                return CGTransform(A.express(CahillGlauberS.val, False))
            
            # In the following A has the same s-value as the transform,
            # so we can simply discard the braces and replace the operators
            # by the corresponding phase-space variables, since the CG transform
            # of an s-ordered operator is a straightforward replacement.
            return op2sc(A.args[0])
        
        def treat_function(A : sp.Function) -> sp.Expr:
            if A.has(densityOp):
                return make(A)

            # The CG transform of any function in only one of 'annihilateOp'
            # and 'createOp' can be evaluated by replacing the 'Operator' with
            # the 'Scalar' counterpart.
            #
            # This also applies to any function in only one of `qOp` and `pOp`,
            # but only in the 's=0' case (the Wigner transform).
            
            def is_evaluable(AA : sp.Function) -> bool:
                sub_found = []
                for oper in AA.atoms(Operator):
                    if oper.sub in sub_found:
                        return False
                    sub_found.append(oper.sub)
                return True
            
            if (is_evaluable(A)
                or (CahillGlauberS.val==0
                    and is_evaluable(alpha2qp(A)))):
                return op2sc(A)
            
            return make(A)
        
        def treat_HattedStar(A : HattedStar) -> sp.Expr:
            return sp.Mul(*mp_helper(A.args, CGTransform))
        
        def treat_der(A : sp.Derivative):
            return sp.Derivative(CGTransform(A.args[0]), *A.args[1:])
            
        def make(A : sp.Expr):
            return super(CGTransform, cls).__new__(cls, A, *_vars)
            
        expr = qp2alpha(sp.sympify(expr))
        if not(expr.has(Operator)) and not(isinstance(expr, NotAScalar)):
            return expr
        return operation_routine(expr,
                                CGTransform,
                                [],
                                [],
                                {(PhaseSpaceVariableOperator, densityOp) : expr},
                                {sp.Add : treat_add,
                                 sp.Mul : treat_mul,
                                 sp.Function : treat_function,
                                 (PhaseSpaceVariableOperator, sp.Pow) : lambda A: op2sc(A),
                                 densityOp : (pi.val)**get_N() * W,
                                 sOrdering : treat_sOrdering,
                                 iCGTransform : lambda A: A.args[0],
                                 Commutator : lambda A: CGTransform(A.doit().expand()),
                                 HattedStar : treat_HattedStar,
                                 sp.Derivative : treat_der}
                                )
        
    def _latex(self, printer):
        return r"\mathcal{W}_{s={%s}}\left[{%s}\right]" % (sp.latex(CahillGlauberS.val),
                                                           sp.latex(self.args[0]))
    
###

class iCGTransform(sp.Expr, HilbertSpaceObject, Defined, NotAScalar):
    @staticmethod
    def _definition():
        lhs = sp.Symbol(_iCGT_str(r"f\left(\bm{\alpha}\right)"))
        rhs = sp.Symbol(r"\int_{\mathbb{R}^{2N}} \frac{\mathrm{d}\bm{\alpha}}{\pi^N} f\left(\bm{\alpha}\right) \mathcal{T}_{s}\left(\bm{\alpha}\right), \quad %s" % _kernel_string)
        return sp.Equality(lhs, rhs)
    definition = _definition()
    
    def __new__(cls, expr : sp.Expr, lazy=False, *_vars) -> sp.Expr:
        if expr.is_Equality:
            return sp.Equality(iCGTransform(expr.lhs, lazy, *_vars),
                               iCGTransform(expr.rhs, lazy, *_vars))
            
        if not(_vars):
            _vars = sub_cache._get_alphaType_oper()
        
        def treat_add(A : sp.Add) -> sp.Expr:
            return sp.Add(*mp_helper(A.args, iCGTransform))
        
        def treat_der(A : sp.Derivative) -> sp.Expr:
            return _treat_der_template(A, alpha, alphaD)
        
        def treat_pow(A : sp.Pow) -> sp.Expr:
            if (isinstance(A.args[0], sp.Derivative)
                or isinstance(A.args[0], StateFunction)):
                return make(A)
            return sc2op(A)            
            
        def treat_mul(A : sp.Mul) -> sp.Expr:
            if not(A.has(StateFunction)):
                return sOrdering(sc2op(A), lazy=lazy)
            
            # Here we assume that W only appears once in a given term. That is,
            # no complex terms like W*Derivative(W, alpha). If these are found,
            # then we keep it unevaluated, since there would be two
            # un-CBopp-able 'StateFunction's to deal with. Might as well not waste
            # resources.
            
            arg_no_W = sp.Number(1)
            arg_with_W = sp.Number(1)
            for arg in A.args:
                if arg.has(StateFunction):
                    if arg_with_W.has(W):
                        return make(A)
                    arg_with_W *= arg
                else:
                    arg_no_W *= arg
            
            # NOTE: This collection process should generally make evaluations
            # faster as we only need to call 'dStar' once.
            
            arg_no_W_iCG = express(sOrdering(sc2op(arg_no_W)), 1, True)
                        # Any 't' produces equally many terms except on some edge cases, so we
                        # choose 't=1' quite arbitrarily, though it may prove to be useful
                        # in normal-ordered cases, which should be the most popular out of the three.
            arg_with_W_iCG = iCGTransform(arg_with_W)
            
            return HattedStar(arg_no_W_iCG, arg_with_W_iCG)
            
        def treat_foo(A: sp.Function) -> sp.Expr:
            if A.has(StateFunction):
                return make(A)
            return sOrdering(sc2op(A))
        
        def treat_Star(A : Star) -> sp.Expr:
            return sp.Mul(*mp_helper(A.args, iCGTransform))
        
        def make(A : sp.Expr) -> iCGTransform:
            return super(iCGTransform, cls).__new__(cls, A, *_vars)
        
        expr = qp2alpha(sp.sympify(expr))
        return operation_routine(expr, 
                                iCGTransform,
                                [],
                                [Operator],
                                {(PhaseSpaceVariable, StateFunction) : expr},
                                {sp.Add : treat_add,
                                 sp.Derivative : treat_der,
                                 sp.Pow : treat_pow,
                                 sp.Mul : treat_mul,
                                 PhaseSpaceVariable : lambda A: sc2op(A),
                                 StateFunction : rho/(pi.val)**get_N(),
                                 sp.Function : treat_foo,
                                 CGTransform : lambda A: A.args[0],
                                 Star : treat_Star}
                                )
    def _latex(self, printer):
        return r"\mathcal{W}^{-1}_{s={%s}}\left[{%s}\right]" % (sp.latex(CahillGlauberS.val),
                                                                sp.latex(self.args[0]))
###

def _make_eq(lhs:str, rhs:str):
    return sp.Equality(sp.Symbol(lhs), sp.Symbol(rhs))

def _property_0():
        lhs = _CGT_str(r"\rho", True)
        rhs = r"\pi^N W_s\left(\bm{\alpha}\right)" # = \left(2\pi\hbar\right)^N W_s\left(\bm{q},\bm{p}\right)"
        rhs += r"\quad\Rightarrow\quad \int_{\mathbb{R}^{2N}} \mathrm{d}\bm{\alpha}\, W_s\left(\bm{\alpha}\right) = 1"
        # rhs += r"=\int_{\mathbb{R}^{2N}} \frac{\mathrm{d}\bm{q}\,\mathrm{d}\bm{p}}{\left(2\hbar\right)^N} W_s\left(\bm{q},\bm{p}\right) = 1"
        return _make_eq(lhs, rhs)
    
def _property_1():
        lhs = _CGT_str(r"\mu\hat{f}+\nu\hat{g}")
        rhs = r"\mu" + _CGT_str(r"\hat{f}") + "+" + r"\nu" + _CGT_str(r"\hat{g}") + r", \quad"
        rhs += _iCGT_str(r"\mu f+\nu g")
        rhs += r"=\mu" + _iCGT_str(r"f") + "+" + r"\nu" + _iCGT_str(r"g")
        rhs += r",\quad \mu,\nu\in\mathbb{C}"
        return _make_eq(lhs, rhs)

class CGTransformProperties:
    
    def __init__(self):
        self.desclist = ["CG transform of the density matrix.",
                         "Complex-linearity of the transforms.",]
        
        self.proplist = [globals()[f"_property_{j}"]() for j in range(2)]
    
    def legend(self):
        out  = "Transform properties of the Cahill-Glauber s-parameterized transform \n"
        out += "="*(len(out)-2) + "\n"
        for j, desc in enumerate(self.desclist):
            out += f"[{j}] {desc} \n"
        return out
    
    def __getitem__(self, key):
        print(self.desclist[key])
        print("=" * len(self.desclist[key]))
        return self.proplist[key]
    
    def __str__(self):
        return self.legend()

    def __repr__(self):
        return self.legend()