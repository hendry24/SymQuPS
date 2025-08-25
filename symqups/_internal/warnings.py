import warnings
from contextlib import contextmanager

@contextmanager
def suppress_warning_repeats():
    warned = set() 
    original_warn = warnings.warn

    def custom_warn(message, *args, **kwargs):
        if message not in warned:
            warned.add(message)
            original_warn(message, *args, **kwargs)

    warnings.warn = custom_warn
    try:
        yield
    finally:
        warnings.warn = original_warn