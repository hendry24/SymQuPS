import sympy as sp
import sympy.physics.quantum as spq
import warnings
from typing import Tuple

from ._internal.basic_routines import operation_routine
from ._internal.grouping import UnBoppable, UnDualBoppable, PhaseSpaceObject, qpType, alphaType, PhaseSpaceVariable, AndClass
from ._internal.cache import Bopp_r_dict, Bopp_l_dict, dBopp_dict
from ._internal.multiprocessing import mp_helper

from .objects.scalars import _Primed, W, _DerivativeSymbol
from .objects.operators import Operator, _CommutatorSymbol

from .manipulations import qp2alpha, _symb2der, _subs_template

# The Star Product
##################

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
    
    .Bopp : Bopp shift the input expression. 
    
    """

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

def _star_base(A : sp.Expr, B : sp.Expr) -> sp.Expr:
    """
    Assumptions:
    - A and B does contains (alpha, alphaD).
    """
    
    if not(A.has(PhaseSpaceObject) or 
        (B.has(PhaseSpaceObject))):
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
        X = B*A
    else:
        A = _eval_Bopp(A, left=False)
        B = _Primed(B)
        X = A*B

    return _symb2der(X).doit().expand()

# Dual Star-Product
###################

def _symb2comm(expr : sp.Expr) -> sp.Expr:
    # We put this here insteaad of 'manipulations' since we only use this here. 
    
    def treat_add(A : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, _symb2comm))
    
    def fico(A : sp.Expr) -> None | Tuple[int, Operator, int|sp.Integer]: # first index and commutator order. 
        def treat_mul(AA : sp.Mul) -> tuple:
            for idx, arg in enumerate(AA.args):
                if isinstance(arg, _CommutatorSymbol):
                    return idx, arg.left, 1
                if arg.has(_CommutatorSymbol):
                    return idx, arg.args[0].left, arg.args[1]
        
        return operation_routine(A,
                                "_symb2comm.fico",
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
    
    return operation_routine(expr,
                            "_symb2der",
                            [],
                            [],
                            {_CommutatorSymbol : expr},
                            {sp.Add : treat_add,
                            (sp.Mul, sp.Pow, _CommutatorSymbol) : replace_comm}
                            )

def _eval_dBopp(expr: sp.Expr) -> sp.Expr:
    return _subs_template(expr, dBopp_dict, (AndClass(Operator, qpType),
                                             AndClass(Operator, alphaType)))

class dBopp(sp.Expr, UnDualBoppable):
    def __new__(cls, expr : sp.Expr) -> sp.Expr:
        expr = sp.sympify(expr)
        
        if expr.has(UnDualBoppable):
            return super().__new__(cls, expr)
        
        res = _eval_dBopp(expr)
        
        return res
    
    def _latex(self, printer) -> str:
        return r"\widetilde{\mathrm{Bopp}}\left[{%s}\right]" % sp.latex(self.args[0])
    
def _dual_star_base(A : sp.Expr, B : sp.Expr) -> sp.Expr:
    
    if any(not(obj.has(Operator)) for obj in [A,B]):
        return A*B
    
    cannot_dBopp_A = A.has(UnDualBoppable) or not(A.is_polynomial(Operator))
    cannot_dBopp_B = B.has(UnDualBoppable) or not(A.is_polynomial(Operator))
    
    if cannot_dBopp_A and cannot_dBopp_B:
        raise _CannotBoppFlag()
    
    if cannot_dBopp_A:
        return _symb2comm(dBopp(B)*A)
    else:
        return _symb2comm(dBopp(A)*B)
    
class dStar(sp.Expr, UnDualBoppable):
    def __new__(cls, *args) -> sp.Expr:
        if not(args):
            return sp.Integer(1)
        
        undboppable_args = []
        
        out = sp.Integer(1)
        for k,arg in enumerate(args):
            try:
                if arg.has(qpType):
                    arg = qp2alpha(arg)
                out = _dual_star_base(out, arg)
            except _CannotBoppFlag:
                if out != 1:
                    undboppable_args.append(out)
                out = arg
                if k == (len(args)-1):
                    undboppable_args.append(arg)
                    
        if undboppable_args:
            return super().__new__(cls, *undboppable_args)
    
        return out
    
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\widetilde{\star}_s \left({%s}\right)" % sp.latex(arg)
        return out