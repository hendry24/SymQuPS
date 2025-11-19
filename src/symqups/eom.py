import sympy as sp
from typing import Sequence

from ._internal.grouping import _AddOnlyExpr
from ._internal.math import separate_operator
from ._internal.preprocessing import preprocess_class

from .objects.scalars import t
from .objects.operators import rho, Operator, TimeDependentOp

from .manipulations import dagger, Commutator, Derivative

from . import hbar

__all__ = ["LindbladMasterEquation"]

class _LindbladDissipator(_AddOnlyExpr):
    def __new__(cls, coef = 1, operator_1 = 1, operator_2 = None):        
        operator_2 = operator_2 if (operator_2 is not None) else operator_1        
        
        if not(operator_1.has(Operator)) and not(operator_2.has(Operator)):
            return sp.Integer(0)
        
        return super().__new__(cls, coef, operator_1, operator_2)
    
    @property
    def coef(self):
        return self.args[0]
    
    @property
    def operator_1(self):
        return self.args[1]
    
    @property
    def operator_2(self):
        return self.args[2]
    
    def _latex(self, printer):
        if self.operator_1 == self.operator_2:
            op_str = sp.latex(self.operator_1)
        else:
            op_str = r"{%s},{%s}" % (sp.latex(self.operator_1), sp.latex(self.operator_2))

        return r"{{%s}\mathcal{D}\left({%s}\right)\left[\rho\right]}" \
                % (sp.latex(self.coef) if (self.coef != 1) else "", 
                   op_str)
    
    def define(self):
        P = self.operator_1
        Q = self.operator_2
        Qd = dagger(Q)
        
        out = (2*P*rho*Qd - rho*Qd*P - Qd*P*rho)
        coef_mul = self.coef / 2
        with sp.evaluate(False): # force pretty printing
            out = coef_mul * out
        
        return out

###

class _LindbladMasterEquation(sp.Equality):
    def __new__(cls, lhs, rhs, **options):
        return super().__new__(cls, lhs, rhs, **options)

@preprocess_class
class LindbladMasterEquation(sp.Basic):
    """
    The Lindblad master equation.
    """
    
    is_Equality = True
        
    def __new__(cls, H : sp.Expr = sp.Integer(0), *dissipators):
        """
        Construct a Lindblad master equation.
        
        Parameters
        ----------
        
        H : sp.Expr, optional
            The Hamiltonian as a function of `qOp`, `pOp`, `annihilateOp` or `createOp`.
            
        *dissipators
            The Lindblad dissipators. Each entry is a `sympy.Expr` or a sequence of length
            2 or 3. 
            
            -   A single `Expr`, is taken as the whole collapse operator. 
            -   For a length-2 sequence, the first entry is the dissipator rate coefficient, 
                while the second entry is the dissipator operator. 
            -   For a length-3 sequence, the first entry is the dissipator rate coefficient,
                while the second and third entries are the two operators which specifies a 
                nondiagonal dissipator. 
        
        """
        lhs = Derivative(TimeDependentOp(rho), t())
        
        dissip_lst = []
        for dissip in dissipators:
            if isinstance(dissip, Sequence):
                match len(dissip):
                    case 2:
                        coef, oper_1 = dissip
                        oper_2 = oper_1
                    case 3:
                        coef, oper_1, oper_2 = dissip
                    case default:
                        msg = "Invalid sequence dissipator specified length."
                        msg += f"Must be either 2 or 3, but got {len(dissip)}."
                        raise ValueError(msg)
            else:
                sqrt_coef, oper_1 = separate_operator(dissip)
                coef = sqrt_coef**2
                oper_2 = oper_1
            
            dissip_obj = _LindbladDissipator(coef, oper_1, oper_2)
            if dissip_obj == 0:
                continue
            dissip_lst.append(dissip_obj)
            
        rhs = sp.Add(-sp.I/hbar.val * Commutator(H, rho),
                     *dissip_lst)
        
        return _LindbladMasterEquation(lhs, rhs)

class LME(LindbladMasterEquation):
    pass