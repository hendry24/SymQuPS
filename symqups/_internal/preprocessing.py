import sympy as sp
import functools
from collections.abc import Mapping
from sympy.core.function import UndefinedFunction
from pickle import PicklingError

def preprocess_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [sp.sympify(arg) for arg in args]
        kwargs = {k : sp.sympify(v) for k,v in kwargs.items()}
        
        # HACK: The multiprocessing implementation has an issue
        # where the first call in a session involving objects
        # defined within that session will raise a Pickling error
        # that does not occur when we call the second time.
        
        try:
            return func(*args, **kwargs)
        except PicklingError:
            return func(*args, **kwargs)

    return wrapper

def preprocess_class(cls):    
    for name, attr in vars(cls).items():
        if callable(attr):
            setattr(cls, name, preprocess_func(attr))
    return cls