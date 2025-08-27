import sympy as sp
import warnings

from ._internal.grouping import UnBoppable, PhaseSpaceObject, qpType, PhaseSpaceVariable
from ._internal.cache import Bopp_r_dict, Bopp_l_dict

from .objects.scalars import _Primed, W

from .manipulations import qp2alpha, _symb2der, _subs_template

###

class _placeholderDot(sp.Expr):
    def _latex(self, printer):
        return r""

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
        
        return _symb2der(res*_placeholderDot(*_Primed(W).args))
                        # No `_Primed` objects facing the user. 

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

class _CannotBoppFlag(BaseException):
    pass

class Star(sp.Expr, UnBoppable):
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
    
    .bopp : Bopp shift the input expression. 
    
    """

    def __new__(cls, *args) -> sp.Expr:
        if not(args):
            return sp.Integer(1)
        
        unboppable_args = []
        
        out = 1
        for k, arg in enumerate(args):
            try:
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
        else:
            return out
        
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\star \left({%s}\right)" % sp.latex(arg)
        return out

def _star_base(A : sp.Expr, B : sp.Expr) -> sp.Expr:
        
    A = sp.sympify(A)
    B = sp.sympify(B)

    if not(A.has(PhaseSpaceObject) or 
        (B.has(PhaseSpaceObject))):
        return A*B

    cannot_Bopp_A = A.has(UnBoppable) or not(A.is_polynomial(PhaseSpaceObject))
    cannot_Bopp_B = B.has(UnBoppable) or not(B.is_polynomial(PhaseSpaceObject))

    if cannot_Bopp_A and cannot_Bopp_B:
        raise _CannotBoppFlag()
    
    if A.has(qpType):
        A = qp2alpha(A)
    if B.has(qpType):
        B = qp2alpha(B)
    # Bopp-shifting functions of (q,p) results in more terms, so we do this for efficiency
    # also. 
    
    if cannot_Bopp_A:
        A = _Primed(A)
        B = _eval_Bopp(B, left=True)
        X = sp.expand(B * A)
    else:
        A = _eval_Bopp(A, left=False)
        B = _Primed(B)
        X = sp.expand(A * B)
                
    return _symb2der(X).doit().expand()