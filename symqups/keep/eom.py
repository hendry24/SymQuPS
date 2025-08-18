import sympy as sp
import sympy.physics.quantum as spq
from functools import cached_property

from ..transforms.wigner_transform import WignerTransform
from ..objects import scalars
from ..objects.operators import densityOp, Dagger
from ..utils.algebra import collect_by_derivative, derivative_not_in_num

__all__ = ["LindbladMasterEquation"]
    
class _LindbladDissipator(_AddOnlyExpr):
    def __new__(cls, rate = 1, operator_1 = 1, operator_2 = None):
        rate = sp.sympify(rate)
        
        operator_1 = sp.sympify(operator_1)
        
        operator_2 = operator_2 if (operator_2 is not None) else operator_1
        operator_2 = sp.sympify(operator_2)
        
        return super().__new__(cls, rate, operator_1, operator_2)
    
    @property
    def rate(self):
        return self.args[0]
    
    @property
    def operator_1(self):
        return self.args[1]
    
    @property
    def operator_2(self):
        return self.args[2]
    
    def __str__(self):
        if self.operator_1 == self.operator_2:
            op_str = sp.latex(self.operator_1)
        else:
            op_str = r"{%s},{%s}" % (sp.latex(self.operator_1), sp.latex(self.operator_2))

        return r"{{%s}\mathcal{D}\left({%s}\right)\left[\rho\right]}" \
                % (sp.latex(self.rate), op_str)
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)
    
    def expand(self):
        rho = densityOp()
        P = self.operator_1
        
        Q = self.operator_2
        Qd = Dagger(Q)
        
        out = (2*P*rho*Qd - rho*Qd*P - Qd*P*rho)
        rate_mul = self.rate / 2
        with sp.evaluate(False): # force pretty printing
            out =  rate_mul * out
        return out
    
class LindbladMasterEquation(sp.Basic):
    """
    The Lindblad master equation. 
    
    Parameters
    ----------
    
    """
    neat_display = True
    
    def __new__(cls, 
                H : sp.Expr,
                dissipators : list[list[sp.Expr, sp.Expr]] = []):
        H = sp.sympify(H)
        dissipators = sp.sympify(dissipators)
        return super().__new__(cls, H, dissipators)
    
    @property
    def H(self):
        return self.args[0]
    
    @cached_property
    def dissipators(self):
        out = []
        for inpt in self.args[1]:
            if len(inpt) == 2:
                inpt.append(None) # or inpt[1], same outcome
            elif len(inpt) == 3:
                pass
            else:
                raise ValueError(r"Invalid dissipator specifier : {%s}"
                                 % (inpt))
            rate, operator_1, operator_2 = inpt
            out.append(_LindbladDissipator(rate=rate, 
                                          operator_1=operator_1, 
                                          operator_2=operator_2))
        return out
    
    @property
    def lhs(self):
        return sp.Derivative(densityOp(), scalars.t())

    @property
    def rhs(self):
        out = -sp.I/scalars.hbar * spq.Commutator(self.H, densityOp())
        for dissip in self.dissipators:
            out += dissip
        return out
    
    @cached_property
    def wigner_transform(self):
        lhs = sp.Derivative(scalars.W(), scalars.t())
        rhs = WignerTransform(self.rhs.doit().expand())
                                            # By calling expand, we effectively call .expand of _LindbladDissipator

            # Collect first to reduce the number of terms. 
        if self.neat_display:
            rhs = derivative_not_in_num(collect_by_derivative(rhs, lhs.args[0]))
        return sp.Equality(lhs, rhs)
    
    def __str__(self):
        return sp.latex(sp.Equality(self.lhs, self.rhs))
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)