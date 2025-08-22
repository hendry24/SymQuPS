import sympy as sp

class _AutoSortedUniqueList(list):
    def __init__(self, *args):
        if args:
            raise ValueError("Must be empty on initialization.")
        super().__init__(*args)
    
    def _append(self, item):
        
        if item in self:
            return 

        super().append(item)
        self.sort(key=sp.default_sort_key)
        
        from ..objects import scalars
        from ..objects.scalars import StateFunction, t, alpha, alphaD
        scalars.W = StateFunction(t(), 
                                  *[cls(sub) for sub in self for cls in (alpha, alphaD)])
        
    def append(self, item):
        raise NotImplementedError
        
    def extend(self, iterable):
        raise NotImplementedError

    def insert(self, index, item):
        raise NotImplementedError

    def __setitem__(self, index, value):
        raise NotImplementedError

    def __iadd__(self, other):
        raise NotImplementedError
          
global sub_cache
sub_cache = _AutoSortedUniqueList()