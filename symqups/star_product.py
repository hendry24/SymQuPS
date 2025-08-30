import sympy as sp
from typing import Tuple
import warnings

from ._internal.basic_routines import operation_routine
from ._internal.grouping import UnBoppable, PhaseSpaceVariable, Defined, qpType
from ._internal.cache import Bopp_r_dict, Bopp_l_dict
from ._internal.multiprocessing import mp_helper

from .objects.base import Base

from .manipulations import qp2alpha, _subs_template

###

class _Primed(Base):
    def _get_symbol_name_and_assumptions(cls, A):
        return r"{%s}'" % sp.latex(A), {"commutative" : False}

        
    def __new__(cls, A : sp.Expr):
        def make(A : sp.Expr):
            return super(_Primed, cls).__new__(cls, A)
        
        def prime_expr(A : sp.Expr):
            return A.subs({X:_Primed(X) for X in A.atoms(PhaseSpaceVariable)})    
        
        return operation_routine(A,
                                 _Primed,
                                 [],
                                 [],
                                 {_Primed, prime_expr},
                                 {PhaseSpaceVariable : make}
                                 )
    
    @property
    def base(self):
        return self._custom_args[0]

###

class _DerivativeSymbol(Base):
    
    def _get_symbol_name_and_assumptions(cls, phase_space_variable):
        return r"\partial_{%s}" % sp.latex(phase_space_variable), {"commutative":False}
    
    def __new__(cls, phase_space_variable : PhaseSpaceVariable):
        def make(A):
            return super(_DerivativeSymbol, cls).__new__(cls, _Primed(A))
            
        return operation_routine(phase_space_variable,
                                 _DerivativeSymbol,
                                 [],
                                 [],
                                 {},
                                 {PhaseSpaceVariable : make}
                                 )
    @property
    def diff_var(self):
        return self._custom_args[0]
    
def _deprime(expr : sp.Expr) -> sp.Expr:
    subs_dict = {X : X.base for X in expr.atoms(_Primed)}
    return expr.subs(subs_dict)

###

def _symb2der(expr : sp.Expr) -> sp.Expr:
    """
    Convert derivative expressions with the package's `_DerivativeSymbol` objects into an equivalent
    expression using `sympy.Derivative'. All `_Primed` variables are then returned to their original
    version.
    """
    
    def treat_add(A : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, _symb2der))

    def fido(A : sp.Expr | _DerivativeSymbol) -> None | Tuple[int, _Primed, int|sp.Integer]:
        """
        Short for "first index and diff order", this function looks for the index in `expr.args`
        which contains `_DerivativeSymbol`, then return that index alongisde the differentiation
        variable (a `_Primed` object) and the differentiation order given in that index (the power
        of `_DerivativeSymbol'). 
        """

        # Everything to the right of the first "derivative operator" symbol
        # must be ordered in .args since we have specified the noncommutativity
        # of the primed symbols. It does not matter if the unprimed symbols get
        # stuck in the middle since the operator does not work on them. What is 
        # important is that _Primed objects are correctly placed with respect to the
        # derivative operators.
        
        def treat_mul(AA : sp.Mul) -> tuple:
            for idx, arg in enumerate(AA.args): 
                if isinstance(arg, _DerivativeSymbol):
                    return idx, arg.diff_var, 1
                if arg.has(_DerivativeSymbol):
                    return idx, arg.args[0].diff_var, arg.args[1]
                    
        return operation_routine(A,
                                fido,
                                [sp.Add],
                                [],
                                {_DerivativeSymbol : None},    # stops the recursion in _der2symb
                                {_DerivativeSymbol : lambda A: (0, A.diff_var, 1),
                                sp.Pow : lambda A: (0, A.args[0].diff_var, A.args[1]),
                                sp.Mul : treat_mul}
                                )
        
    def replace_diff(A : sp.Expr) -> sp.Expr:
        """
        Recursively replace the differential operator symbols,
        with the appropriate `sympy.Derivative` objects. Input must
        not be Add.
        """
        
        fido_res = fido(A)

        if fido_res: # no more recursion if fido is None
            cut_idx, diff_var, diff_order = fido_res
            prefactor = A.args[:cut_idx]
            A_leftover = sp.Mul(*A.args[cut_idx+1:])
            return sp.Mul(*prefactor,
                            sp.Derivative(replace_diff(A_leftover),
                                        *[diff_var]*diff_order))
            
            # With this code, we can afford to replace any power of the first
            # _DerivativeSymbol we encounter, instead of replacing only the base
            # and letting the rest of the factors be dealt with in the next recursion
            # node, making the recursion more efficient. 
        
        return A

    return _deprime(operation_routine(expr,
                                      replace_diff,
                                      [],
                                      [],
                                      {_DerivativeSymbol : expr},
                                      {sp.Add : treat_add,
                                       (sp.Mul, sp.Pow, _DerivativeSymbol) : replace_diff}
                                      )
    )

class Bopp(sp.Expr, UnBoppable):
    """
    Bopp shift the input quantity for the calculation of the Moyal star-product. 

    `A(q,p)★B(q,p) = A(q + (i*hbar/2)*dpp, p - (i*hbar/2)*dqq) * B(qq, pp)`

    `A(x,p)★B(x,p) = B(q - (i*hbar/2)*dpp, p + (i*hbar/2)*dqq) * A(qq, pp)`
    
    In the current version, this operation attempts to remove all `sympy.Derivative`
    in the input expression to ensure a correct Bopp shift. This is why `Star` only
    accepts one 'UndefinedFunction' at maximum, as the algorithm can just Bopp-shift
    the other derivative-free expression. 
            
    Parameters
    ----------

    A : sympy.Expr
        Quantity to be Bopp-shifted, should contain `objects.q` or `objects.p`.

    left : bool, default: True
        Whether the star-product operator is to the `left` of `A`. The resulting 
        Bopp-shifted expression contains differential operators that act on
        the other expression situated to the `left` of the star-product operator.

    Returns
    -------

    out : sympy object
        Bopp-shifted sympy object. 

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p

    """
        
    def __new__(cls, expr : sp.Expr, left : bool = False) -> sp.Expr:
        
        expr = sp.sympify(expr)
        
        if expr.has(sp.Derivative):
            expr = expr.doit()
            if expr.has(sp.Derivative):
                msg = "'A' contains persistent derivative(s), most possibly working on an UndefinedFunction. "
                msg += "The function has tried to call 'A.doit()' but could not evaluate the 'Derivative' "
                msg += "objecs. This may lead to faulty Bopp-shifting which results in incorrect ★-products evaluation. "
                msg += "As such, nothing is done to the input and a generic expression is returned."
                warnings.warn(msg)
                return super().__new__(cls, expr, left)
            
        if expr.has(UnBoppable):
            return super().__new__(cls, expr, left)
            
        """
        The derivative evaluation attempt in `Bopp` will deal with intermediate
        unevaluated derivative(s) during the ★-product chain in `Star`. Since
        `Bopp` is not called on the operand containing an UndefinedFunction, this
        effectively keeps the derivative(s) operating on expressions containing
        the UndefinedFunction from being evaluated, resulting in a "prettier" output.
        
        The evaluation-prohibition is not applied in the current version, but the above code
        is nevertheless useful to catch errors, so we keep it there.
        """
        res = _eval_Bopp(expr, left)
        
        return res

    def _latex(self, printer) -> str:
        bopp_arrow = r"\longrightarrow"
        if self.args[1]:
            bopp_arrow = r"\longleftarrow"
        return r"\overset{%s}{\mathrm{Bopp}}\left[{%s}\right]" % (bopp_arrow, sp.latex(self.args[0]))

def _eval_Bopp(expr : sp.Expr, left : bool) -> sp.Expr:
    """
    Evaluae Bopp-shift of `expr`, assuming that it is Boppable.
    """
    Bopp_dict = Bopp_r_dict
    if left:
        Bopp_dict = Bopp_l_dict
        
    return _subs_template(expr, Bopp_dict, (PhaseSpaceVariable,))

###

class _CannotBoppFlag(BaseException):
    pass

def _star_base(A : sp.Expr, B : sp.Expr) -> sp.Expr:
    """
    Assumptions:
    - A and B does contains (alpha, alphaD).
    """
    
    if not(A.has(PhaseSpaceVariable) or B.has(PhaseSpaceVariable)):
        return A*B

    cannot_Bopp_A = A.has(UnBoppable) or not(A.is_polynomial(PhaseSpaceVariable))
                                        # should also be true if 'A" contains 'Derivative' objects, which
                                        # raises an error in 'Bopp' if unevaluable. 
    cannot_Bopp_B = B.has(UnBoppable) or not(B.is_polynomial(PhaseSpaceVariable))

    if cannot_Bopp_A and cannot_Bopp_B:
        raise _CannotBoppFlag()
    
    if cannot_Bopp_A:
        A = _Primed(A)
        B = _eval_Bopp(B, left=True)
        X = sp.expand(B*A)
    else:
        A = _eval_Bopp(A, left=False)
        B = _Primed(B)
        X = sp.expand(A*B)

    return _symb2der(X).doit()

class Star(sp.Expr, UnBoppable, Defined):
    """
    The s-parameterized star-product `A(q,p) ★ B(q,p) ★ ...` (or the `alpha` equivalent), 
    calculated using the Bopp shift.

    Parameters
    ----------

    *args
        The factors of the star-product, ordered from first to last. Since the algorithm
        utilizes the Bopp shift, only one operand be a non-polynomial.

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p
    
    See Also
    --------
    
    .Bopp : Bopp shift the input expression. 
    
    """

    @staticmethod
    def _definition():
        out = r"f\left(\bm{\alpha},\overline{\bm{\alpha}}\right)"
        out += r"\mathbin{\star_s}"
        out += r"g\left(\bm{\alpha},\overline{\bm{\alpha}}\right)"
        out += r"= f\left(\bm{\alpha}+\frac{s+1}{2}\partial_{\bm{\alpha}'},"
        out += r"\overline{\bm{\alpha}} + \frac{s-1}{2}\partial_{\overline{\bm{\alpha}}'}\right)"
        out += r"g\left(\bm{\alpha}',\overline{\bm{\alpha}}'\right)"
        return sp.Symbol(out)
    definition = _definition()
    
    def __new__(cls, *args) -> sp.Expr:
        if not(args):
            return sp.Integer(1)
        
        unboppable_args = []
        
        out = sp.Integer(1)
        for k, arg in enumerate(args):
            try:
                if arg.has(qpType):
                    arg = qp2alpha(arg)
                out = _star_base(out, arg)
            except _CannotBoppFlag:
                if out != 1:
                    unboppable_args.append(out)
                out = arg
                if k == (len(args)-1):
                    unboppable_args.append(arg)
        
        if unboppable_args:
            # msg = "One or more pairs of consecutive inputs cannot be properly "
            # msg += "Bopp-shifted to work with the package. "
            # msg += "In the current version, non-polynomial inputs are unboppable."
            # pprint(msg)
            return super().__new__(cls, *unboppable_args)
        
        return out
        
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\star_s \left({%s}\right)" % sp.latex(arg)
        return out