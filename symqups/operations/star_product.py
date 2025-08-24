import sympy as sp
from pprint import pprint

from ..objects.base import PhaseSpaceObject, qpTypePSO
from ..objects import scalars
from ..objects.scalars import q, p, alpha, alphaD, _DerivativeSymbol, _Primed, _deprime
from ..utils.multiprocessing import _mp_helper
from .._internal.basic_routines import invalid_input
from ..utils.algebra import qp2a
from .. import s

__all__ = ["Bopp",
           "Star"]

class Bopp(sp.Expr):
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
        
    def __new__(cls, expr : sp.Expr, left : bool = False):
        
        expr = sp.sympify(expr)
        
        if expr.has(sp.Derivative):
            expr = expr.doit()
            if expr.has(sp.Derivative):
                msg = "'A' contains persistent derivative(s), most possibly working on an UndefinedFunction. "
                msg += "The function has tried to call 'A.doit()' but could not evaluate the 'Derivative' "
                msg += "objecs. This may lead to faulty Bopp-shifting which results in incorrect ★-products evaluation. "
                msg += "As such, nothing is done to the input and a generic expression is returned."
                pprint(msg)
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
        
        def dxx(X):
            return _DerivativeSymbol(_Primed(X))

        sgn = 1
        if left:
            sgn = -1
        
        hbar = scalars.hbar
        mu = scalars.mu
        
        subs_dict = {}
        for A in list(expr.atoms(PhaseSpaceObject)):
            if isinstance(A, alpha):
                subs_dict[A] = A + (s.val + sgn)/2*dxx(alphaD(A.sub))
            elif isinstance(A, alphaD):
                subs_dict[A] = A + (s.val - sgn)/2*dxx(alpha(A.sub)) 
            elif isinstance(A, q):
                subs_dict[A] = A + sgn*sp.I*hbar/2*dxx(p(A.sub)) + (s.val*hbar/2)*(1/mu**2)*dxx(q(A.sub))
            elif isinstance(A, p):
                subs_dict[A] = A - sgn*sp.I*hbar/2*dxx(q(A.sub)) + (s.val*hbar/2)*(mu**2)*dxx(p(A.sub))
            else:
                 invalid_input(A, "Bopp")
                        
        return sp.expand(expr.subs(subs_dict))
    
    def _latex(self, printer):
        left = self.args[1]
        star_latex = r"\star"
        bopp_arg = r"\left({%s}\right)" % sp.latex(self.args[0])
        if left:
            bopp_arg = star_latex + bopp_arg
        else:
            bopp_arg = bopp_arg + star_latex
        return r"\mathrm{Bopp}\left[{%s}\right]" % bopp_arg

class _CannotBoppFlag(BaseException):
    pass

class Star(sp.Expr):
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

    def __new__(cls, *args):
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
            msg = "One or more pairs of consecutive inputs cannot be properly "
            msg += "Bopp-shifted to work with the package. "
            msg += "In the current version, non-polynomial inputs are unboppable."
            pprint(msg)
            return super().__new__(cls, *unboppable_args)
        else:
            return out
        
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\star \left({%s}\right)" % sp.latex(arg)
        return out

def _star_base(A : sp.Expr, B : sp.Expr) \
    -> sp.Expr:
        
    A = sp.sympify(A)
    B = sp.sympify(B)

    if not(A.has(PhaseSpaceObject) or 
        (B.has(PhaseSpaceObject))):
        return A*B

    cannot_Bopp_A = not(A.is_polynomial(scalars.Scalar))
    cannot_Bopp_B = not(B.is_polynomial(scalars.Scalar))

    if cannot_Bopp_A and cannot_Bopp_B:
        raise _CannotBoppFlag()
    
    if A.has(qpTypePSO):
        A = qp2a(A)
    if B.has(qpTypePSO):
        B = qp2a(B)
    # Bopp-shifting functions of (q,p) results in more terms, so we do this for efficiency
    # also. 
    
    if cannot_Bopp_A:
        A = scalars._Primed(A)
        B = Bopp(B, left=True)
        X = sp.expand(B * A)
    else:
        A = Bopp(A, left=False)
        B = scalars._Primed(B)
        X = sp.expand(A * B)

    # Expanding is necessary to ensure that all arguments of X contain no Add objects.
    #
    # The ★-product evaluation routine called after Bopp shifting, whence
    # the primed objects are no longer needed. This function loops through
    # the arguments of the input `X` (generally an `Add` object) and replaces 
    # the primed objects by the appropriate, functional Objects, i.e., the unprimed
    # variables and `sympy.Derivative`. For the derivative objects, this is recursively 
    # done by `_replace_diff`. This function then replaces q' and p' by q and p, respectively.
    
    if isinstance(X, sp.Add):
        X_args = X.args
    else:
        X_args = [X]
    
    out = sp.Add(*_mp_helper(X_args, _replace_diff))
                
    return _deprime(out).doit().expand()

def _first_index_and_diff_order(A : sp.Expr) \
    -> None | tuple[int, scalars.q|scalars.p, int|sp.Number]:
    """
    
    Get the index of the first differential operator appearing
    in the Bopp-shifted expression (dqq or dpp), either qq or pp, and 
    the differential order (the power of dqq or dpp).
    
    Parameters
    ----------
    
    A : sympy.Expr
        A summand in the expanded Bopp-shifted expression to be
        evaluated. `A.args` thus give its factors.
        
    Returns
    -------
    
    idx : int
        The index of `A.args` where the first `_DerivativeSymbol` object is contained.
        
    diff_var : `qq` or `pp`
        The primed differentiation variable. Either `qq` or `pp`, accessed by taking the 
        `.diff_var` attribute of the `_DerivativeSymbol`, returning the `_Primed` 
        object. It stays _Primed here since the other factors in the Expr that
        the derivative is supposed to work on is the ones containing _Primed.
        
    diff_order : int or sp.Number
        The order of the differentiation contained in the `idx`-th argument of 
        `A`, i.e., the exponent of `_DerivativeSymbol` encountered.
    """

    # Everything to the right of the first "derivative operator" symbol
    # must be ordered in .args since we have specified the noncommutativity
    # of the primed symbols. It does not matter if the unprimed symbols get
    # stuck in the middle since the operator does not work on them. What is 
    # important is that x' and p' are correctly placed with respect to the
    # derivative operators.
    
    A = A.expand()
    if isinstance(A, sp.Add):
        raise TypeError("Input must not be 'Add'.")
    
    if not(A.has(_DerivativeSymbol)):
        return None # This stops the recursion. See _replace_diff.
    
    if isinstance(A, _DerivativeSymbol):
        return 0, A.diff_var, 1

    if isinstance(A, sp.Pow):
        # We have dxx**n for n>1. For a Pow object, the second argument gives
        # the exponent; in this case, the differentiation order.
        return 0, A.args[0].diff_var, A.args[1]
    
    if isinstance(A, sp.Mul):
        for idx, A_ in enumerate(A.args): 
            if isinstance(A_, _DerivativeSymbol):
                return idx, A_.diff_var, 1
            if A_.has(_DerivativeSymbol):
                return idx, A_.args[0].diff_var, A_.args[1]
                
    raise TypeError(r"Invalid input: \n\n {%s}" % sp.latex(A))

def _replace_diff(A : sp.Expr) -> sp.Expr:
    """
    Recursively replace the differential operator symbols,
    with the appropriate `sympy.Derivative` objects. Here _Primed 
    objects stay as is for _star_base to differentiate correctly.
    
    Parameters
    ----------
    
    A : sympy.Expr
        Expression generally containing _DerivativeSymbols, as well as _Primed
        and functions thereof. 
    """
    
    fido = _first_index_and_diff_order(A)

    if fido: # no more recursion if fido is None
        cut_idx, diff_var, diff_order = fido
        prefactor = A.args[:cut_idx]
        A_leftover = sp.Mul(*A.args[cut_idx+1:])
        return sp.Mul(*prefactor,
                        sp.Derivative(_replace_diff(A_leftover),
                                      *[diff_var]*diff_order))
        
        # With this code, we can afford to replace any power of the first
        # dqq or dpp we encounter, instead of replacing only the base
        # and letting the rest of the factors be dealt with in the next recursion
        # node, making the recursion more efficient. 
    
    return A