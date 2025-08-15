import sympy as sp
from typing import Tuple, Callable

class Base(sp.Symbol):
    """
    Base object for the package, essentially a modified sympy.Symbol supporting extra accessible
    arguments. 
    """
    
    def _get_symbol_name_and_assumptions(cls, *custom_args):
        raise NotImplementedError()
    
    def __new__(cls, *custom_args):
        
        name, assumptions = cls._get_symbol_name_and_assumptions(cls, *custom_args)
        
        obj = super().__new__(cls,
                              name = name,
                              **assumptions)
        obj._custom_args = custom_args 
        """
        '_args' is used by SymPy and should not be overriden, or the method
        .atoms which is crucial in this package will not work correctly.
            
        This allows us to store custom_args as accessible attributes. We can
        also set what each custom argument is called in a given subclass, 
        by defining a property then returning the argument.
        """        
        return obj

    def __reduce__(self):
        # This specifies how pickling is done for the object and its subclasses.
        # .assumptions0 is needed by sympy.Symbol
        # See https://docs.python.org/3/library/pickle.html
        return self.__class__, self._custom_args, self.assumptions0
    
class _Set(set):
    def update(self, *args, **kwargs):
        s = "This object should not be modified by the user. "
        s += "Call the '_update' method to force-update the object."
        raise AttributeError(s)
    
    def _update(self, *args, **kwargs):
        super().update(*args, **kwargs)
global _sub_cache
_sub_cache = _Set([])

def _treat_sub(sub, has_sub):
    if ((sub is None) or not(has_sub)):
        return sp.Symbol(r"")
    if isinstance(sub, str):
        return sp.Symbol(sub)
    if isinstance(sub, sp.Symbol):
        return sub
    return sp.Symbol(sp.latex(sub))

def _screen_type(expr : sp.Expr, forbidden_type : object, name : str):
    if expr.has(forbidden_type):
        msg = f"'{name}' does not accept '{forbidden_type}'"
        raise TypeError(msg)

def _invalid_input(inpt : object, name : str):
    msg = f"Invalid input to '{name}':\n"
    msg += r"%s" % sp.latex(inpt)
    raise ValueError(msg)

def _operation_routine(expr : sp.Expr,
                       name : str,
                       forbidden_types : tuple[type],
                       if_expr_does_not_have : tuple[type],
                       return_if_expr_does_not_have : Callable[[sp.Expr], sp.Expr],
                       *return_if_expr_is : tuple[Tuple[tuple[type], Callable[[sp.Expr], sp.Expr]]]):
    
    expr = sp.expand(sp.sympify(expr))
    
    _screen_type(expr, forbidden_types, name)
    
    if not(expr.has(*if_expr_does_not_have)):
        return return_if_expr_does_not_have(expr)
    
    for if_expr_is, then_return in return_if_expr_is:
        if isinstance(expr, if_expr_is):
            return then_return(expr)
        
    _invalid_input(expr, name)