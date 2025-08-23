import os
from multiprocessing import Pool
import dill
from functools import partial

from typing import TypedDict, Sequence

############################################################

__all__ = ["MP_CONFIG"]

############################################################
global _mp_is_running
_mp_is_running = False

# NOTE: Same code as in pybolano.

class _mp_dict(TypedDict):
    enable: bool
    num_cpus: int
    min_num_args: int

    def __setitem__(self, key, value):
        valid_keys = ["enable", "num_cpus", "min_num_args", "_in_use"]
        if key not in valid_keys:
            msg = f"The key [{key}] is not valid. Valid keys: {valid_keys}."
            raise KeyError(msg)

        if key == "min_num_args":
            if value >= 2:
                super().__setitem__(key, value)
            else:
                super().__setitem__(key, 2)
        else:
            super().__setitem__(key, value)

############################################################

MP_CONFIG = _mp_dict()
MP_CONFIG["enable"] = True
MP_CONFIG["num_cpus"] = os.cpu_count()
MP_CONFIG["min_num_args"] = 2
# Skip multiprocessing if the number of elements is spall,
# in which case a single core execution is enough.

def _mp_helper(inpt_lst : Sequence, foo : callable):
    """
    Apply `foo` to a sequence of inputs, using multiprocessing
    if possible.
    """
    global _mp_is_running
    
    use_mp = (not(_mp_is_running) and 
            MP_CONFIG["enable"] and 
            (len(inpt_lst) >= MP_CONFIG["min_num_args"]))
    
    if use_mp:
        _mp_is_running = True
        with Pool(MP_CONFIG["num_cpus"]) as pool:
            res_byte_lst = pool.map(partial(_pool_helper, foo=foo),
                                    [dill.dumps(inpt) for inpt in inpt_lst])
        _mp_is_running = False
        return [dill.loads(res_bytes) for res_bytes in res_byte_lst]
    else:
        return [foo(inpt) for inpt in inpt_lst]
    
def _pool_helper(inpt_bytes : bytes, foo : callable):
    """
    The package usage involves `sympy.Function`, which the
    package `pickle`, used by `multiprocessing`, cannot pickle.
    As a workaround, here we use `dill` to pickle everything before 
    sending the job to the worker processes. This is the topmost
    function called by a worker process, which loads the bytes input
    by the main process, reconstructing the SymPy objects for 
    `_replace_diff` to work with. Then, the output is pickled once 
    again when sent back to the main process. 
    """
    
    return dill.dumps(foo(dill.loads(inpt_bytes)))