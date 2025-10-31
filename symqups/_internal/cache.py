import sympy as sp

class SubCache(list):
    def __init__(self, *args) -> None:
        if args:
            raise ValueError("Must be empty on initialization.")
        super().__init__(*args)
    
    def _append(self, sub) -> None:
        
        if sub in self:
            return

        super().append(sub)
        self.sort(key=sp.default_sort_key)
        
        self._refresh(sub)
        
    def _refresh(self, sub):
        
        from ..objects.scalars import t, q, p, alpha, alphaD, W 
        from ..objects.operators import qOp, pOp, annihilateOp, createOp
        
        from .. import zeta, hbar
        zeta = zeta.val
        hbar = hbar.val
        
        ###
        
        W._set_args((t(), *[cls(s) for s in self for cls in (alpha, alphaD)]))
        
        ###
        
        qq = q(sub)
        pp = p(sub)
        a = alpha(sub)
        ad = alphaD(sub)
        
        qop = qOp(sub)
        pop = pOp(sub)
        aop = annihilateOp(sub)
        adop = createOp(sub)
        
        ###
        
        for k, v in [[qq, qop],
                     [pp, pop],
                     [a, aop],
                     [ad, adop]]:
            sc2op_subs_dict._set_item(k, v)
            op2sc_subs_dict._set_item(v, k)
        
        # NOTE: These are "naive" substitution dictionaries, 
        # so no substituting W with rho and vice versa.
        
        ###
        
        for k, v in [[qq, sp.sqrt(hbar)/sp.sqrt(2) * (1/zeta) * (a + ad)],
                     [pp, sp.sqrt(hbar)/sp.sqrt(2) * (zeta/sp.I) * (a - ad)]]:
            qp2alpha_subs_dict._set_item(k, v)
            qp2alpha_subs_dict._set_item(sc2op_subs_dict[k], 
                                         v.xreplace({a:aop, ad:adop}))

        ###
        
        for k, v in [[a, (zeta*qq + sp.I*(1/zeta)*pp)/sp.sqrt(2*hbar)],
                     [ad, (zeta*qq - sp.I*(1/zeta)*pp)/sp.sqrt(2*hbar)]]:
            alpha2qp_subs_dict._set_item(k, v)            
            alpha2qp_subs_dict._set_item(sc2op_subs_dict[k], 
                                         v.xreplace({qq:qop, pp:pop}))
            
    def _refresh_all(self):
        for sub in self:
            self._refresh(sub)
            
    def _get_alphaType_scalar(self):
        from ..objects.scalars import alpha, alphaD
        return [cls(sub) for sub in self for cls in [alpha, alphaD]]
    
    def _get_alphaType_oper(self):
        from ..objects.operators import annihilateOp, createOp
        return [cls(sub) for sub in self for cls in [annihilateOp, createOp]]
        
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
    
class ProtectedDict(dict):
    def __setitem__(self, key, value):
        raise NotImplementedError("No touching.")
    
    def _set_item(self, key, value):
        super().__setitem__(key, value)
              
global op2sc_subs_dict, sc2op_subs_dict, qp2alpha_subs_dict
global alpha2qp_subs_dict, sub_cache
qp2alpha_subs_dict = ProtectedDict()
alpha2qp_subs_dict = ProtectedDict()
op2sc_subs_dict = ProtectedDict()
sc2op_subs_dict = ProtectedDict()

sub_cache = SubCache()