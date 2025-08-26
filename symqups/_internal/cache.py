import sympy as sp

from .. import zeta, hbar

class AutoSortedUniqueList(list):
    def __init__(self, *args) -> None:
        if args:
            raise ValueError("Must be empty on initialization.")
        super().__init__(*args)
    
    def _append(self, sub) -> None:
        
        if sub in self:
            return

        super().append(sub)
        self.sort(key=sp.default_sort_key)
        
        ###
        
        from ..objects.scalars import t, q, p, alpha, alphaD, W, _Primed, _DerivativeSymbol
        from ..objects.operators import qOp, pOp, annihilateOp, createOp, rho
        
        W._set_args((t(), *[cls(s) for s in self for cls in (alpha, alphaD)]))
        
        ###
        
        def der(x):
            return _DerivativeSymbol(_Primed(x))
        
        qq = q(sub)
        pp = p(sub)
        a = alpha(sub)
        ad = alphaD(sub)
        
        qop = qOp(sub)
        pop = pOp(sub)
        aop = annihilateOp(sub)
        adop = createOp(sub)
        
        for k, v in [[qq, qop],
                     [pp, pop],
                     [a, aop],
                     [ad, adop]]:
            sc2op_subs_dict._set_item(k, v)
            op2sc_subs_dict._set_item(k, v)
        
        for k, v in [[qq, sp.sqrt(hbar.val/2) * (1/zeta.val) * (a + ad)],
                     [pp, sp.sqrt(hbar.val/2) * (zeta.val/sp.I) * (a - ad)]]:
            qp2alpha_subs_dict._set_item(k, v)
            qp2alpha_subs_dict._set_item(sc2op_subs_dict[k], 
                                         v.subs({a:aop, ad:adop}))
        
        for k, v in [[a, (zeta.val*qq + sp.I*(1/zeta.val)*pp)/sp.sqrt(2*hbar.val)],
                     [ad, (zeta.val*qq - sp.I*(1/zeta.val)*pp)/sp.sqrt(2*hbar.val)]]:
            alpha2qp_subs_dict._set_item(k, v)
            alpha2qp_subs_dict._set_item(sc2op_subs_dict[k], 
                                         v.subs({qq:qop, pp:pop}))
            
        # NOTE: {rho : W} substitution rule is set in 'object.Operator'.
        
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
        super()[key] = value
              
global op2sc_subs_dict, sc2op_subs_dict, qp2alpha_subs_dict, alpha2qp_subs_dict, sub_cache
qp2alpha_subs_dict = ProtectedDict()
alpha2qp_subs_dict = ProtectedDict()
op2sc_subs_dict = ProtectedDict()
sc2op_subs_dict = ProtectedDict()

sub_cache = AutoSortedUniqueList()