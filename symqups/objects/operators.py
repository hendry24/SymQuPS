import sympy as sp

import typing
from . import scalars 
# Avoid, e.g., "from scalars import hbar" 
# since we want its value to follow changes at runtime.
from .base import Base
from .cache import _sub_cache
from ..utils._internal_routines import _treat_sub, _operation_routine
from ..utils.multiprocessing import _mp_helper

class Operator(Base):
    
    base = NotImplemented
    has_sub = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        return r"%s_{%s}" % (cls.base, sub), {"commutative":False}
    
    def __new__(cls, sub = None):
        sub = _treat_sub(sub, cls.has_sub)
        
        global _sub_cache
        _sub_cache._update([sub])
        
        return super().__new__(cls, sub)
        
    @property
    def sub(self):
        return self._custom_args[0]
    
    def dagger(self):
        raise NotImplementedError()
    
    def wigner_transform(self):
        raise NotImplementedError()

class Dagger():
    """
    Hermitian conjugate of `A`.
    """
    def __new__(cls, expr : sp.Expr | Operator):
        return _operation_routine(expr,
                                  "Dagger",
                                  (),
                                  (Operator,),
                                  lambda A: A.conjugate(),
                                  (Operator,
                                  lambda A:  A.dagger()),
                                  (sp.Add, 
                                   lambda A: sp.Add(*_mp_helper(A.args, Dagger))), 
                                  (sp.Pow,
                                   lambda A: Dagger(A.args[0]) ** A.args[1]),
                                  (sp.Mul,
                                   lambda A: sp.Mul(*list(reversed(_mp_helper(A.args, Dagger)))))
                                  )
    
class HermitianOp(Operator):
    @typing.final
    def dagger(self):
        return self
    
class qOp(HermitianOp):
    base = r"\hat{q}"
    
class pOp(HermitianOp):
    base = r"\hat{p}"
    
class annihilateOp(Operator):
    base = r"\hat{a}"
        
    def define(self):
        with sp.evaluate(False):
            return ((qOp(sub=self.sub)*scalars.mu + sp.I*pOp(sub=self.sub)/scalars.mu) 
                    / sp.sqrt(2*scalars.hbar))
    
    def dagger(self):
        return createOp(sub = self.sub)
    
class createOp(Operator):
    base = r"\hat{a}^{\dagger}"
    
    def define(self):
        with sp.evaluate(False):
            mu_conj = sp.conjugate(scalars.mu)
            
            return ((qOp(sub=self.sub)*mu_conj - sp.I*pOp(sub=self.sub)/mu_conj) 
                    / sp.sqrt(2*scalars.hbar))
        
    def dagger(self):
        return annihilateOp(sub = self.sub)
    
class densityOp(HermitianOp):
    base = r"\rho"
    has_sub = False
    
    def __new__(cls, sub=None):
        return super().__new__(cls, sub)
    
class rho():
    def __new__(cls):
        return densityOp()