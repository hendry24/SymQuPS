class _Set(set):
    def update(self, *args, **kwargs):
        s = "This object should not be modified by the user. "
        s += "Call the '_update' method to force-update the object."
        raise AttributeError(s)
    
    def _update(self, *args, **kwargs):
        super().update(*args, **kwargs)
global _sub_cache
_sub_cache = _Set([])