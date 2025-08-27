import sympy as sp

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
        
        self._refresh(sub)
        
    def _refresh(self, sub):
        
        from ..objects.scalars import t, q, p, alpha, alphaD, W, _Primed, _DerivativeSymbol
        from ..objects.operators import qOp, pOp, annihilateOp, createOp
        from .. import zeta, hbar, s
        
        zeta = zeta.val
        hbar = hbar.val
        CGs = s.val
        
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
        
        def der(x):
            return _DerivativeSymbol(_Primed(x))
        
        ###
        
        for k, v in [[qq, sp.sqrt(hbar)/sp.sqrt(2) * (1/zeta) * (a + ad)],
                     [pp, sp.sqrt(hbar)/sp.sqrt(2) * (zeta/sp.I) * (a - ad)]]:
            qp2alpha_subs_dict._set_item(k, v)
            qp2alpha_subs_dict._set_item(_Primed(k), _Primed(v))
            
            qp2alpha_subs_dict._set_item(sc2op_subs_dict[k], 
                                         v.subs({a:aop, ad:adop}))
            
            da_dqq = zeta / sp.sqrt(2*hbar)
            dad_dqq = da_dqq
            da_dpp = (sp.I/sp.sqrt(2*hbar)) * (1/zeta)
            dad_dpp = -da_dpp
            
            qp2alpha_subs_dict._set_item(der(qq), 
                                         da_dqq*der(a) + dad_dqq*der(ad))
            qp2alpha_subs_dict._set_item(der(pp), 
                                         da_dpp*der(a) + dad_dpp*der(ad))

        ###
        
        for k, v in [[a, (zeta*qq + sp.I*(1/zeta)*pp)/sp.sqrt(2*hbar)],
                     [ad, (zeta*qq - sp.I*(1/zeta)*pp)/sp.sqrt(2*hbar)]]:
            alpha2qp_subs_dict._set_item(k, v)
            alpha2qp_subs_dict._set_item(_Primed(k), _Primed(v))
            
            alpha2qp_subs_dict._set_item(sc2op_subs_dict[k], 
                                         v.subs({qq:qop, pp:pop}))
            
            dqq_da = sp.sqrt(hbar)/sp.sqrt(2) * (1/zeta)
            dqq_dad = dqq_da
            dpp_da = sp.sqrt(hbar)/sp.sqrt(2) * (zeta/sp.I)
            dpp_dad = -dpp_da
            
            alpha2qp_subs_dict._set_item(der(a),
                                         dqq_da*der(qq) + dpp_da*der(pp))
            alpha2qp_subs_dict._set_item(der(ad),
                                         dqq_dad*der(qq) + dpp_dad*der(pp))
            
        ###
        
        for Bopp_dict, sgn in [[Bopp_r_dict, 1],
                               [Bopp_l_dict, -1]]:
            Bopp_dict : ProtectedDict
            
            Bopp_dict._set_item(qq, qq 
                                    + sgn * sp.I*hbar/2 * der(pp) 
                                    + (CGs*hbar/2) * (1/zeta**2) * der(qq))
            Bopp_dict._set_item(pp, pp
                                    - sgn * sp.I*hbar/2 * der(qq)
                                    + (CGs*hbar/2) * zeta**2 * der(pp))
            Bopp_dict._set_item(a,  a 
                                    + (CGs+sgn)/2 * der(ad))
            Bopp_dict._set_item(ad, ad
                                    + (CGs-sgn)/2 * der(a))
            
    def _refresh_all(self):
        for sub in self:
            self._refresh(sub)
        
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
global alpha2qp_subs_dict, Bopp_r_dict, Bopp_l_dict, sub_cache
qp2alpha_subs_dict = ProtectedDict()
alpha2qp_subs_dict = ProtectedDict()
op2sc_subs_dict = ProtectedDict()
sc2op_subs_dict = ProtectedDict()
Bopp_r_dict = ProtectedDict()
Bopp_l_dict = ProtectedDict()

sub_cache = AutoSortedUniqueList()