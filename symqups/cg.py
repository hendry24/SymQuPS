import sympy as sp
import sympy.physics.quantum as spq
from typing import Tuple

from ._internal.multiprocessing import mp_helper, MP_CONFIG
from ._internal.basic_routines import operation_routine
from ._internal.grouping import PhaseSpaceVariable, PhaseSpaceObject, Defined, HilbertSpaceObject
from ._internal.cache import sub_cache

from .objects.base import Base
from .objects.scalars import W, StateFunction, alpha, alphaD
from .objects.operators import Operator, densityOp, rho, annihilateOp, createOp

from .star_product import Star
from .ordering import sOrdering
from .manipulations import qp2alpha, op2sc, alpha2qp, sc2op, dagger, express
from .utils import get_N

from . import s as CahillGlauberS
from . import pi, hbar

###

_kernel_string = r"\Delta_s\left(\bm{\alpha}\right) = "
_kernel_string += r"\int_{\mathbb{R}^{2N}}\mathrm{d}\bm{\beta}"
_kernel_string += r"\exp\left(-\frac{1+s}{1-s}\left|\bm{\beta}\right|^2\right)"
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

class CGTransform(sp.Expr, PhaseSpaceObject, Defined):
    
    @staticmethod
    def _definition():
        lhs = sp.Symbol(_CGT_str(r"\hat{A}"))
        rhs = sp.Symbol(r"\mathrm{tr}\left(\hat{A}\Delta_s\left(\bm{\alpha}\right)\right), \quad %s" % _kernel_string)
        return sp.Equality(lhs, rhs)
    definition = _definition()
    
    def __new__(cls, expr : sp.Expr):
        """
        oper -> quantum ps vars
        """
        
        def treat_add(A : sp.Expr) -> sp.Expr:
            return sp.Add(*mp_helper(A.args, CGTransform))
        
        def treat_substitutable(A : sp.Expr) -> sp.Expr:
            if isinstance(A, densityOp):
                return (pi.val)**get_N() * W
            return op2sc(A)
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
            return Star(*mp_helper(A.args, CGTransform))
            
        def treat_sOrdering(A : sOrdering) -> sp.Expr:
            if (A.args[1] != CahillGlauberS.val):
                if not(A.args[0].is_polynomial(Operator)):
                    return make(A)
                return CGTransform(A.express(CahillGlauberS.val))
            
            # In the following A has the same s-value as the transform,
            # so we can simply discard the braces and replace the operators
            # by the corresponding phase-space variables, since the CG transform
            # of an s-ordered operator is a straightforward replacement.
            return treat_substitutable(A.args[0])
        
        def treat_function(A : sp.Function) -> sp.Expr:
            if rho in A.atoms(Operator):
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
                return treat_substitutable(A)
            
            return make(A)
            
        def make(A : sp.Expr):
            return super(CGTransform, cls).__new__(cls, A)
            
        expr = qp2alpha(sp.sympify(expr.doit()))  # '.doit' in case there is Commutator.
        return operation_routine(expr,
                                "CG_transform",
                                [],
                                [PhaseSpaceVariable],
                                {Operator : expr},
                                {sp.Add : treat_add,
                                sp.Mul : treat_mul,
                                (Operator, sp.Pow) : treat_substitutable,
                                sp.Function : treat_function,
                                sOrdering : treat_sOrdering})
        
    def _latex(self, printer):
        return r"\mathcal{W}_{s={%s}}\left[{%s}\right]" % (sp.latex(CahillGlauberS.val),
                                                           sp.latex(self.args[0]))
    
###

class iCGTransform(sp.Expr, HilbertSpaceObject, Defined):
    @staticmethod
    def _definition():
        lhs = sp.Symbol(_iCGT_str(r"f\left(\bm{\alpha}\right)"))
        rhs = sp.Symbol(r"\int \frac{\mathrm{d}\bm{\alpha}}{\pi^N} f\left(\bm{\alpha}\right) \Delta_{-s}\left(\bm{\alpha}\right), \quad %s" % _kernel_string)
        return sp.Equality(lhs, rhs)
    definition = _definition()
    
    def __new__(cls, expr : sp.Expr, lazy=False) -> sp.Expr:
        def treat_add(A : sp.Add) -> sp.Expr:
            return sp.Add(*mp_helper(A.args, iCGTransform))
        
        def treat_der(A : sp.Derivative) -> sp.Expr:
            A = A.doit()
            if not(isinstance(A, sp.Derivative)):
                return iCGTransform(A)
                        
            der_args = list(A.args)
            
            diff_var = der_args[1][0] # leftmost derivative.
            if isinstance(diff_var, alpha):
                comm_left = -createOp(diff_var.sub)
            else:
                comm_left = annihilateOp(diff_var.sub)
            
            der_args[1] = (diff_var, der_args[1][1] - 1) 
            # sp.Derivative(f, (x, 0)) is valid and returns f.
            # When this happens, the recursion into 'treat_der" stops.
            return spq.Commutator(comm_left, iCGTransform(sp.Derivative(*der_args)))
        
        def treat_pow(A : sp.Pow) -> sp.Expr:
            if (isinstance(A.args[0], sp.Derivative)
                or isinstance(A.args[0], StateFunction)):
                return make(A)
            return sc2op(A)            
            
        def treat_mul(A : sp.Mul) -> sp.Expr:
            if not(A.has(StateFunction)):
                return treat_substitutable(A)
            
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
                    
            # We now do an analogue of the Bopp shift where we have commutator
            # brackets instead of differenial operators, which we call the 
            # 'CBopp' (Commutator-Bopp). Since sympy requires the argument to
            # be specified on instantiation, we again turn the "commutator bracket
            # operator" into a noncommuting symbol we can freely multiply.  
            #
            # NOTE: Unlike the operator version, we can freely move the scalars
            # around so we can always CBopp from the left.
            
            class _CommutatorSymbol(Base, HilbertSpaceObject):
                def _get_symbol_name_and_assumptions(cls, left):
                    return r"\left[%s, \cdot\right]" % (sp.latex(left)), {"commutative" : False}

                def __new__(cls, left : sp.Expr):
                    return super().__new__(cls, left)
                
                @property
                def left(self):
                    return self._custom_args[0]
            
            subs_dict = {}
            for a in arg_no_W.atoms(alpha):
                # Here we skip the naive sc2op substitution. The naive substitution plus
                # the CBopp is similar to how we compute the 'CGTransform' of an operator
                # and then 'Bopp'-shifting the resulting phase-space function.
                aop = annihilateOp(a.sub)
                subs_dict[aop] = aop - (CahillGlauberS.val + 1)/2 * _CommutatorSymbol(aop)
                # NOTE: Yes, 'annihilateOp' in '_CommutatorSymbol', not 'createOp'.
            for ad in arg_no_W.atoms(alphaD):
                adop = createOp(ad.sub)
                subs_dict[adop] = adop - (1 - CahillGlauberS.val)/2 * _CommutatorSymbol(adop)
            
                            # This is the 'iCGTransform' of arg_no_W
            arg_no_W_iCG = express(sOrdering(sc2op(arg_no_W)), 1, True).subs(subs_dict)
                        # Any 't' produces equally many terms except on some edge cases, so we
                        # choose 't=1' quite arbitrarily, though it may prove to be useful
                        # in normal-ordered cases, which should be the most popular out of the three.
            
            X = sp.expand(arg_no_W_iCG * iCGTransform(arg_with_W))
                                    # No need to '_Primed' this since operators are noncommutative.
            
            # We now recursively replace '_CommutatorSymbol' by the real thing.
            # Code is similar to 'fido' in symqups.manipulations. Though it is interesting
            # to combine the two, it is too much work and may not be trivial.
            def fico(A : sp.Expr) -> None | Tuple[int, Operator, int|sp.Integer]: # first index and commutator order. 
                def treat_mul(AA : sp.Mul) -> tuple:
                    for idx, arg in enumerate(AA.args):
                        if isinstance(arg, _CommutatorSymbol):
                            return idx, arg.left, 1
                        if arg.has(_CommutatorSymbol):
                            return idx, arg.args[0].left, arg.args[1]
                
                return operation_routine(A,
                                         "iCGTransform.__new__.treat_mul.fico",
                                         [sp.Add],
                                         [],
                                         {_CommutatorSymbol : None},
                                         {_CommutatorSymbol : lambda A: (0, A.left, 1),
                                          sp.Pow : lambda A: (0, A.args[0].left, A.args[1]),
                                          sp.Mul : treat_mul}
                                         )
            
            def replace_comm(A : sp.Expr) -> sp.Expr:
                fico_res = fico(A)
                
                if fico_res:
                    cut_idx, comm_left, comm_order = fico_res
                    prefactor = A.args[:cut_idx]
                    A_leftover = _CommutatorSymbol(comm_left)**(comm_order-1) * sp.Mul(*A.args[cut_idx+1:])
                    # NOTE: This is where it differs from sp.Derivative. We can't just shortcut and do multiple 
                    # commutator brackets. 
                    return sp.Mul(*prefactor,
                                  spq.Commutator(comm_left, replace_comm(A_leftover)))
                    
                return A
            
            if isinstance(X, sp.Add):
                X_args = X.args
            else:
                X_args = [X]
            
            # NOTE: It is a pain to make 'replace_comm' pickleable, so we just run this
            # without multiprocessing, which is generally likely the case.
            original_mp_enable = MP_CONFIG["enable"]
            MP_CONFIG["enable"] = False
            out = sp.Add(*mp_helper(X_args, replace_comm))
            MP_CONFIG["enable"] = original_mp_enable
            return out            
        
        def treat_W(A : StateFunction) -> sp.Expr:
            return rho/(pi.val)**get_N()
        
        def treat_foo(A: sp.Function) -> sp.Expr:
            if A.has(StateFunction):
                return make(A)
            return treat_substitutable(A)
        
        def treat_substitutable(A: sp.Expr) -> sp.Expr:
            return sOrdering(sc2op(A))
        
        def make(A : sp.Expr) -> iCGTransform:
            super(iCGTransform, cls).__new__(cls, A)
        
        expr = qp2alpha(sp.sympify(expr))
        return operation_routine(expr, 
                                "iCG_transform",
                                [],
                                [Operator],
                                {(PhaseSpaceVariable, StateFunction) : expr},
                                {sp.Add : treat_add,
                                 StateFunction : treat_W,
                                 sp.Derivative : treat_der,
                                 sp.Pow : treat_pow,
                                 sp.Mul : treat_mul,
                                 PhaseSpaceVariable : treat_substitutable,
                                 sp.Function : treat_foo}
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