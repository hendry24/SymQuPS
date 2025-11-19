import sympy as sp
import itertools, functools

from ._internal.multiprocessing import mp_helper
from ._internal.basic_routines import operation_routine
from ._internal.grouping import (PhaseSpaceVariable, PhaseSpaceObject, Defined, 
                                 HilbertSpaceObject, NotAnOperator, NotAScalar,
                                 PhaseSpaceVariableOperator)
from ._internal.math import has_universal_oper
from ._internal.cache import sub_cache
from ._internal.preprocessing import preprocess_class

from .objects.scalars import W, StateFunction, alpha, alphaD
from .objects.operators import (Operator, densityOp, rho, annihilateOp, 
                                createOp, TimeDependentOp)

from .bopp import PSBO
from .star import Star, HattedStar
from .ordering import sOrdering
from .manipulations import (qp2alpha, op2sc, alpha2qp, sc2op, Commutator,
                            s_ordered_equivalent, dagger, normal_ordered_equivalent,
                            Derivative)
from .utils import get_N

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

class _CreateOpAsPSV(annihilateOp):
    # Identifies as an annihilateOp but does not trigger 
    # sub-caching. Used in `CGTransform::treat_mul` to 
    # *represent* the PSV part of a PSBO.
    def __new__(cls, psv : alpha | alphaD):
        obj = super(Operator, cls).__new__(cls,
                                           sp.latex(psv.sub)+str(type(psv)))
                                        # need to make the sub different 
                                        # between alpha and alphaD 
                                        # since d(alphaD) and alpha commute,
                                        # etc.
        obj.psv = psv
        return obj
    
class _AnnihilateOpAsDer(createOp):
    # Represents the derivative part of a PSBO.
    def __new__(cls, psv : alpha | alphaD, xi : sp.Expr):
        obj = super(Operator, cls).__new__(cls, 
                                           sp.latex(psv.sub)+str(type(psv)))
        obj.xi = xi
        obj.diff_wrt = dagger(psv)
        return obj
    
###

def _normal_ordered_PSBO_on_B(combo, B):
    # Used in CGTransform::treat_mul::do_when_nonpoly_found.
    rep_word = sp.Mul(*combo)
    rep_word_NO = normal_ordered_equivalent(rep_word).expand()
    
    if isinstance(rep_word_NO, sp.Add):
        rep_word_NO_summands = rep_word_NO.args
    else:
        rep_word_NO_summands = [rep_word_NO]
    
    out_summands = []
    for term in rep_word_NO_summands:
        if term.is_Mul:
            args = term.args
        else:
            args = [term]
            
        coef = []
        xi = []
        psv = []
        diff_B_wrt = []
        for o in args:
            b, e = o.as_base_exp()
            if isinstance(b, _CreateOpAsPSV):
                psv.extend([b.psv]*e)
            elif isinstance(b, _AnnihilateOpAsDer):
                xi.extend([b.xi]*e)
                diff_B_wrt.append((b.diff_wrt, e))
            else:
                coef.append(o)
        
        out_summands.append(sp.Mul(*coef,
                                   *xi,
                                   *psv,
                                   Derivative(B, *diff_B_wrt)))
    
    return sp.Add(*out_summands)

@preprocess_class
class CGTransform(sp.Expr, PhaseSpaceObject, Defined, NotAnOperator):
    
    @staticmethod
    def _definition():
        lhs = sp.Symbol(_CGT_str(r"\hat{A}"))
        rhs = sp.Symbol(r"\mathrm{tr}\left(\hat{A}\mathcal{T}_{-s}\left(\bm{\alpha}\right)\right), \quad %s" % _kernel_string)
        return sp.Equality(lhs, rhs)
    definition = _definition()
    
    def __new__(cls, expr : sp.Expr, *_vars, mode = "Star"):
        """
        
        PARAMETERS
        ----------
        
        mode : str
            Evaluation mode for expressions containing polynomial and nonpolynomial
            parts.
            
            (1) Cascaded application of PSBOs. This is SLOW, possibly due
                to the multiply nested expressions. Running `.doit` 
                afterwards would also be more expensive due to all 
                the nesting. This is mode `"PSBO"`.
            
            (2) Star product of the CG Transforn of the factors 
                divided by polynomiality. This is FAST, since we 
                can abuse the efficiency of the star-product algorithm
                which has no ordering problem for its arguments. 
                This is mode `"Star"`.
            
            (3) Make every possible word where every operator is
                replaced by the corresponding PSBO split into 
                the PSV part and the derivative part. Then, normal
                order each word utilizing the similarity between
                `[aOp, adOp] = 1` and `[dx, x] = 1` to obtain the explicit
                series expanded to the greatest extend possible (i.e., 
                with the product rule already evaluated). This is SLOW,
                because it takes multiple loops to (i) separate the factors
                by polynomiality, (ii) make each word, (iii) normal order
                each word, and (iv) evaluate each term in the output. This
                is mode `"explicit"`.

        """
        
        if isinstance(mode, sp.Symbol): # deal with decorator
            mode = mode.name
        mode = mode.lower()
        if mode not in ["star", "psbo", "explicit"]:
            msg = "Invalid mode. Either 'Star','PSBO', or 'explicit'."
            raise ValueError(msg)
        
        if expr.is_Equality:
            from .eom import _LindbladMasterEquation
            
            lhs = expr.lhs
            rhs = expr.rhs
            if isinstance(expr, _LindbladMasterEquation):
                lhs /= pi.val
                rhs /= pi.val
            return sp.Equality(CGTransform(lhs),
                               CGTransform(rhs))
        
        if not(_vars):
            _vars = sub_cache._get_alphaType_scalar()
        
        def treat_add(A : sp.Expr) -> sp.Expr:
            return sp.Add(*mp_helper(A.args, CGTransform))
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
        
            if (A.is_polynomial(PhaseSpaceVariableOperator) and
                all(isinstance(atom, PhaseSpaceVariableOperator) 
                    for atom in A.atoms(Operator))):
                return op2sc(s_ordered_equivalent(A))
            
            ###
            
            s = CahillGlauberS.val
            
            coefs = []
            out_star_factors = [] 
            poly_factors = []
            nonpoly = None # not used in mode "Star"
            
            def transform_poly_factors(poly_factors):
                return op2sc(s_ordered_equivalent(sp.Mul(*poly_factors)))
            
            def nonpoly_found_PSBO(poly_factors, nonpoly, left):
                out = CGTransform(nonpoly)
                for o in poly_factors:
                    b, e = o.as_base_exp()
                    for _ in range(e):
                        out = PSBO(op2sc(b), out, left)
                return out
            
            def nonpoly_found_explicit(poly_factors, nonpoly, left):
                # left = nonpoly is to the left of poly
                #
                # Here we turn `poly` into a word of PSBOs that
                # operate on whatever is to its operation direction.
                # Unlike the usual Bopp shifts, however, we cannot 
                # just let the PSVs bypass the derivative operator.
                # As such, we need an extra trick to obtain the most
                # explciit series (in comparison to the algorithm used in,
                # say, `Star`).
                #
                # We note that the derivative operator and the PSVs obey
                # a commutation relation [dx, x] = 1 just like the commutation
                # relation [a, ad] = 1. As such, dx can play the role of a, while
                # x can play the role of ad, and what we want to do is to obtain the
                # normal-ordered series. We can thus make use of 
                # `normal_ordered equivalent`. It would be more efficient to rewrite
                # the algorithm here, but for readibility and convenience 
                # let us substitute dx and x with the ladder operators. Since the 
                # objects for ad and a can pass through each other, we can assign 
                # them to ladder operators with different subscripts.
                #
                # Note that dx here is a right-directed derivative, so the LAST
                # entry in `poly_factors` `is the FIRST to apply to the CGTransform
                # of `nonpoly'.
                
                lr_sign = 1
                if left:
                    lr_sign = -1

                xi_a = sp.Rational(1,2) * (s + lr_sign)
                xi_ad = sp.Rational(1,2) * (s - lr_sign)
                
                to_combo = []
                for o in poly_factors:
                    b, e = o.as_base_exp()
                
                    xi = xi_a if isinstance(b, annihilateOp) else xi_ad    
                    psv = op2sc(b)
                    
                    psv_term = _CreateOpAsPSV(psv)
                    der_term = _AnnihilateOpAsDer(dagger(psv), xi)
                    
                    to_combo.extend([[psv_term, der_term]]*e)
                
                return sp.Add(*mp_helper(list(itertools.product(*to_combo)),
                                        functools.partial(_normal_ordered_PSBO_on_B,
                                                        B = CGTransform(nonpoly))))

            nonpoly_found = nonpoly_found_PSBO if mode.lower() == "PSBO" else nonpoly_found_explicit
            
            for arg in A.args:
                arg : sp.Expr
                if arg.has(PhaseSpaceVariableOperator):
                    if arg.is_polynomial(PhaseSpaceVariableOperator):
                        poly_factors.append(arg)
                    else:
                        if mode == "star":
                            out_star_factors.append(op2sc(s_ordered_equivalent(sp.Mul(*poly_factors))))
                            out_star_factors.append(CGTransform(arg))
                        else:
                            if nonpoly is None:
                                nonpoly = nonpoly_found(poly_factors, arg, False)
                            else:
                                out_star_factors.append(nonpoly_found(poly_factors, nonpoly, True))
                                nonpoly = arg
                            
                        poly_factors = []
                else:
                    if arg.has(Operator):
                        # We can avoid code duplicate by using some flags, but
                        # let us put efficiency first here. Since the two are close
                        # together, they would be easy to fix anyway.
                        if mode == "star":
                            out_star_factors.append(transform_poly_factors(poly_factors))
                            out_star_factors.append(CGTransform(arg))
                        else:
                            if nonpoly is None:
                                nonpoly = nonpoly_found(poly_factors, arg, False)
                            else:
                                out_star_factors.append(nonpoly_found(poly_factors, nonpoly, True))
                                nonpoly = arg
                            
                        poly_factors = []
                    else:
                        coefs.append(arg)
                        
            # Loop may end with nonpoly or poly in the operators. 
            # 
            # For mode "Star", if the loop ends with a nonpoly, then we need not
            # do anything. If it ends with a poly, then we CG-transform the leftover
            # polynomial part and add to the Star-factors.
            #
            # For mode "PSBO" or "explicit", if the loop
            # ends with a nonpoly, then we just append that nonpoly into
            # the out_star_factors (poly_factors would be empty since there is no
            # iteration after nonpoly). Otherwise, we apply left-directed PSBOs
            # to the last nonpoly found.
            
            if mode == "star":
                if poly_factors:
                    out_star_factors.append(transform_poly_factors(poly_factors))
            else:
                out_star_factors.append(nonpoly_found(poly_factors, nonpoly, True))
            
            return sp.Mul(*coefs, Star(*out_star_factors))

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
            if has_universal_oper(A):
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
            return Derivative(CGTransform(A.args[0]), *A.args[1:])
        
        def treat_commutator(A : Commutator):
            return CGTransform(A.args[0]*A.args[1] - A.args[1]*A.args[0])
        
        def treat_tdOp(A : TimeDependentOp):
            # Assuming rhoTD only.
            return CGTransform(A.args[0])
            
        def make(A : sp.Expr):
            return super(CGTransform, cls).__new__(cls, A, *_vars)
        
        expr = qp2alpha(expr)
        return operation_routine(expr,
                                CGTransform,
                                [],
                                [PhaseSpaceObject],
                                {(PhaseSpaceVariableOperator, densityOp) : expr},
                                {sp.Add : treat_add,
                                 sp.Mul : treat_mul,
                                 sp.Function : treat_function,
                                 (PhaseSpaceVariableOperator, sp.Pow) : lambda A: op2sc(A),
                                 densityOp : (pi.val)**get_N() * W,
                                 sOrdering : treat_sOrdering,
                                 iCGTransform : lambda A: A.args[0],
                                 Commutator : treat_commutator,
                                 HattedStar : treat_HattedStar,
                                 sp.Derivative : treat_der,
                                 TimeDependentOp : treat_tdOp}
                                )
        
    def _latex(self, printer):
        return r"\mathcal{W}_{s={%s}}\left[{%s}\right]" % (sp.latex(CahillGlauberS.val),
                                                           sp.latex(self.args[0]))
    
###

@preprocess_class
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
            der_args = A.args
            new_der_args = [iCGTransform(der_args[0])]
            for arg in der_args[1:]:
                new_der_args.append((iCGTransform(arg[0]), arg[1]))
            return Derivative(*new_der_args)
        
        def treat_pow(A : sp.Pow) -> sp.Expr:
            if (isinstance(A.args[0], sp.Derivative)
                or isinstance(A.args[0], StateFunction)):
                return make(A)
            return sc2op(A)            
            
        def treat_mul(A : sp.Mul) -> sp.Expr:
            if (A.is_polynomial(PhaseSpaceVariable) and
                not(A.has(StateFunction))):
                return sOrdering(sc2op(A), lazy=lazy)
            
            # TODO: Might want to optimize this part by fully using 
            # the combo method, since there is nested Add-and-Mul
            # for the output generation below. This will increase
            # cost with number of subsystems. May not be needed if
            # use cases is focused on single sub, in which case the
            # explicit sum may be faster than assembling combos.                                                                            
            
            coefs = []
            m_dict = {sub : 0 for sub in sub_cache}
            n_dict = {sub : 0 for sub in sub_cache}
            others = []
            for arg in A.args:
                if arg.has(PhaseSpaceVariable):
                    if arg.is_polynomial(PhaseSpaceVariable):                        
                        b, e = arg.as_base_exp()
                        if isinstance(b, alphaD):
                            m_dict[b.sub] += e
                        elif isinstance(b, alpha):
                            n_dict[b.sub] += e
                        else:
                            raise ValueError("Invalid value here.")
                    else:
                        others.append(arg)
                else:
                    coefs.append(arg)
                        
            # Here, 'others' may contain nonpolynomial functions in
            # the PSV like exp(alpha) and is generally a Mul object.
            # Since it is not really meaningful to make a long string of hatted
            # star products, we shall keep 'others' unevaluated if
            # it is a composite expression like Mul, Pow, etc. Since anything
            # other than unevaluated Mul is already dealt with by other 'treat'
            # functions, we only check if 'others' is a Mul itself to avoid
            # infinite recursions.
            
            # Since the HSBSs commute, we can just write out the explicit
            # series rather than letting sympy do it with the `.doit' method.`
            
            if len(others) == 0:
                F = 1
            elif len(others) == 1:
                F = iCGTransform(others[0])
            else:
                F = make(sp.Mul(*others))
                # Since we can collect all nonpolynomials together, it would avoid
                # clutter to keep the nonpolynomial part unevaluated.
            
            # NOTE: Here we use the explicit form of the cascaded HSBS applications.
            out = sp.Mul(*coefs, F)
            for sub in sub_cache:
                m = m_dict[sub]
                n = n_dict[sub]
                ad = createOp(sub)
                a = annihilateOp(sub)
                k1 = (1-CahillGlauberS.val)/2
                k2 = (1+CahillGlauberS.val)/2
                summands = []
                for j in range(m+1):
                    for k in range(n+1):
                        factors = [sp.binomial(m,j),
                                   sp.binomial(n,k),
                                   k1**(n+j-k),
                                   k2**(m-j+k),
                                   ad**(m-j),
                                   a**(n-k),
                                   out,
                                   a**k,
                                   ad**j
                                   ]
                        summands.append(sp.Mul(*factors))
                out = sp.Add(*summands)
                
            return out
            
        def treat_foo(A: sp.Function) -> sp.Expr:
            if A.has(StateFunction):
                return make(A)
            return sOrdering(sc2op(A))
        
        def treat_Star(A : Star) -> sp.Expr:
            return sp.Mul(*mp_helper(A.args, iCGTransform))
                
        def make(A : sp.Expr) -> iCGTransform:
            return super(iCGTransform, cls).__new__(cls, A, *_vars)
        
        expr = qp2alpha(expr)
        return operation_routine(expr, 
                                iCGTransform,
                                [],
                                [HilbertSpaceObject],
                                {(PhaseSpaceVariable, StateFunction) : expr},
                                {sp.Add : treat_add,
                                 sp.Derivative : treat_der,
                                 sp.Pow : treat_pow,
                                 sp.Mul : treat_mul,
                                 PhaseSpaceVariable : lambda A: sc2op(A),
                                 StateFunction : TimeDependentOp(rho)/(pi.val)**get_N(),
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

class CGProps:
    
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