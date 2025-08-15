import sympy as sp

from . import scalars
from ..utils.multiprocessing import _mp_helper

__all__ = ["Bopp",
           "Star"]

class Bopp():
    """
    Bopp shift the input quantity for the calculation of the Moyal star-product. 

    `A(q,p)★B(q,p) = A(q + (i/2)*dpp, p - (i/2)*dqq) * B(qq, pp)`

    `A(x,p)★B(x,p) = B(q - (i/2)*dpp, p + (i/2)*dqq) * A(qq, pp)`
    
    In the current version, this operation attempts to remove all `sympy.Derivative`
    in the input expression to ensure a correct Bopp shift. This is why `Star` only
    accepts one 'UndefinedFunction' at maximum, as the algorithm can just Bopp-shift
    the other derivative-free expression. 
            
    Parameters
    ----------

    A : sympy.Expr
        Quantity to be Bopp-shifted, should contain `objects.q` or `objects.p`.

    left : bool, default: False
        Whether the star-product operator is to the left of `A`. 

    Returns
    -------

    out : sympy object
        Bopp-shifted sympy object. 

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p

    """
        
    def __new__(cls, A : sp.Expr, left : bool = False):
        
        if A.has(sp.Derivative):
            A = A.doit()
            if A.has(sp.Derivative):
                s = "'A' contains persistent derivative(s), most possibly working on an UndefinedFunction."
                s += "The function has tried to call 'A.doit()' but couldn't get rid of the 'Derivative"
                s += "objecs. This leads to faulty Bopp-shifting which results in incorrect ★-products evaluation."
                raise ValueError(s)
            
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
            return scalars._DerivativeSymbol(scalars._Primed(X))

        sgn = 1
        if left:
            sgn = -1
        
        subs_dict = {}
        for X in list(A.atoms()):
            if isinstance(X, scalars.q):
                subs_dict[X] = X + sgn * sp.I*scalars.hbar/2 * dxx(scalars.p(X.sub))
            if isinstance(X, scalars.p):
                subs_dict[X] = X - sgn * sp.I*scalars.hbar/2 *  dxx(scalars.q(X.sub))
        
        return A.subs(subs_dict).expand()
    
class Star():
    """
    The Moyal star-product A(q,p) ★ B(q,p) ★ ..., calculated using the Bopp shift.

    Parameters
    ----------

    *args
        The factors of the star-product, ordered from first to last. Since the algorithm
        utilizes the Bopp shift, only one operand may be "un-Bopp-shift-able", i.e. which contains:
        (1) `sympy.Function`'s in `q` or `p`, or
        (2) `sympy.Pow`'s that have `q` or `p` in the exponents, or
        (3) `sympy.Pow`'s that are `q` or `p` raised to some non-positive-integer exponent.

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
        
        out = sp.sympify(args[0])
        for arg in args[1:]:
            out = _star_base(out, sp.sympify(arg))
        return out
    
def _star_base(A : sp.Expr, B : sp.Expr) \
    -> sp.Expr:    
    any_phase_space_variable_in_A = A.has(scalars.q, scalars.p)
    any_phase_space_variable_in_B = B.has(scalars.q, scalars.p)
    if (not(any_phase_space_variable_in_A) or 
        not(any_phase_space_variable_in_B)):
        return A*B

    def cannot_Bopp_pow(X):
        """
        Pow is not Function, so it needs a special treatment. Here we prevent
        Bopp shift if the expression contains Pow objects that:
            - Has q or p in the exponents.
            - Is a non-positive-integer power of q or p. 
        """
        pow_in_X = X.find(sp.Pow)
        pow_in_X_with_qp = [x for x in pow_in_X if x.has(scalars.q, scalars.p)]
        for x in pow_in_X_with_qp:
            exp = x.args[1]
            if not(isinstance(exp, sp.Integer) and exp >= 0):
                return True
        return False

    cannot_Bopp_A, cannot_Bopp_B = \
        [any([x.atoms(scalars.q, scalars.p) for x in X.find(sp.Function)])
         or cannot_Bopp_pow(X) for X in [A,B]]

    if cannot_Bopp_A and cannot_Bopp_B:
        msg = "Both inputs cannot be properly Bopp shifted to work with the package. "
        msg += "Expressions that contain: "
        msg += "(1) 'Function's in q or p, or "
        msg += "(2) 'Pow's that have q or p in the exponents, or "
        msg += "(3) 'Pow's that are q or p raised to some non-positive-integer exponent, "
        msg += "are problematic when Bopp-shifted."
        raise ValueError(msg)
    
    if cannot_Bopp_A:
        A = scalars._Primed(A)
        B = Bopp(B, left=True)
        X = (B * A).expand()
    else:
        A = Bopp(A, left=False)
        B = scalars._Primed(B)
        X = (A * B).expand()

    # Expanding is necessary to ensure that all arguments of X contain no Add objects.
    
    """
    The ★-product evaluation routine called after Bopp shifting, whence
    the primed objects are no longer needed. This function loops through
    the arguments of the input `X` (generally an `Add` object) and replaces 
    the primed objects by the appropriate, functional Objects, i.e., the unprimed
    variables and `sympy.Derivative`. For the derivative objects, this is recursively 
    done by `_replace_diff`. This function then replaces q' and p' by q and p, respectively.
    """

    X : sp.Expr
    if isinstance(X, sp.Add):
        X_args = X.args
    else:
        X_args = [X]
    
    out = sp.Add(*_mp_helper(X_args, _replace_diff))
                
    return scalars._DePrimed(out).doit().expand()

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

    """
    Everything to the right of the first "derivative operator" symbol
    must be ordered in .args since we have specified the noncommutativity
    of the primed symbols. It does not matter if the unprimed symbols get
    stuck in the middle since the operator does not work on them. What is 
    important is that x' and p' are correctly placed with respect to the
    derivative operators.
    """
    A = A.expand()
    if isinstance(A, sp.Add):
        raise TypeError("Input must not be 'Add'.")
    
    if not(A.has(scalars._DerivativeSymbol)):
        return None # This stops the recursion. See _replace_diff.
    
    if isinstance(A, scalars._DerivativeSymbol):
        return 0, A.diff_var, 1

    if isinstance(A, sp.Pow):
        # We have dxx**n for n>1. For a Pow object, the second argument gives
        # the exponent; in this case, the differentiation order.
        return 0, A.args[0].diff_var, A.args[1]
    
    if isinstance(A, sp.Mul):
        for idx, A_ in enumerate(A.args): 
            if isinstance(A_, scalars._DerivativeSymbol):
                return idx, A_.diff_var, 1
            if A_.has(scalars._DerivativeSymbol):
                return idx, A_.args[0].diff_var, A_.args[1]
                
    raise TypeError(r"Invalid input: \n\n {%s}" % sp.latex(A))

def _replace_diff(A : sp.Expr) \
    -> sp.Expr:
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
        """
        With this code, we can afford to replace any power of the first
        dqq or dpp we encounter, instead of replacing only the base
        and letting the rest of the factors be dealt with in the next recursion
        node, making the recursion more efficient. 
        """
    
    return A