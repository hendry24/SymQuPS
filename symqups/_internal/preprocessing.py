import sympy as sp
import functools
from pickle import PicklingError

def preprocess_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        
        sympified_args = []
        for arg in args:
            try:
                sympified_args.append(sp.sympify(arg))
            except:
                sympified_args.append(arg)
            
        sympified_kwargs = {}
        for k,v in kwargs.items():
            try:
                sympified_kwargs[k] = sp.sympify(v)
            except:
                sympified_kwargs[k] = v
        
        # HACK: The multiprocessing implementation has an issue
        # where the first call in a session involving objects
        # defined within that session will raise a Pickling error
        # that does not occur when we call the second time.
        
        try:
            return func(*sympified_args, **sympified_kwargs)
        except PicklingError:
            return func(*sympified_args, **sympified_kwargs)

    return wrapper

def preprocess_class(cls):    
    for name, attr in vars(cls).items():
        if callable(attr):
            setattr(cls, name, preprocess_func(attr))
    return cls