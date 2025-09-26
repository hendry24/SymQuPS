import sympy as sp
from typing import Sequence
from functools import cached_property

from ._internal.grouping import _AddOnlyExpr
from ._internal.operator_handling import separate_operator

from .objects.scalars import W, t
from .objects.operators import rho, Operator

from .manipulations import dagger, Commutator
from .cg import CGTransform
from .utils import derivative_not_in_num, collect_by_derivative

from . import hbar, pi

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

class StateFunctionEvo(sp.Equality):
    pass

###

class LindbladMasterEquation(sp.Basic):
    """
    The Lindblad master equation. 
    
    Parameters
    ----------
    
    """
    
    is_Equality = True
    
    def __new__(cls, H : sp.Expr = sp.Integer(0), *dissipators, **options):
        H = sp.sympify(H)
        dissipators = sp.sympify(dissipators)
        
        lhs = sp.Derivative(rho, t())
        
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
        
        obj = super().__new__(cls, lhs, rhs, **options)
        
        obj._H = H
        obj._dissipators = dissip_lst
        obj._lhs = lhs
        obj._rhs = rhs
        obj._equality = sp.Equality(lhs, rhs)
        
        return obj
    
    @property
    def H(self) -> sp.Expr:
        return self._H
    
    @property
    def dissipators(self) -> list:
        return self._dissipators
    
    @property
    def lhs(self) -> sp.Derivative:
        return self._lhs
    
    @property
    def rhs(self) -> sp.Expr:
        return self._rhs
    
    def _latex(self, printer):
        return sp.latex(self._equality)

    @cached_property
    def CG_transform(self):
        lhs = sp.Derivative(W, t())
        
        rhs = self.H.doit()
        for dissip in self.dissipators:
            rhs += dissip.define()
        
        rhs = CGTransform(rhs/pi.val)

        return StateFunctionEvo(lhs, rhs)