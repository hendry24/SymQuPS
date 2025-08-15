

@pytest.mark.order(3)
class TestStarProduct():
    
    rand_N = random.randint(0, 100)
    
    x = sp.Symbol("x")
    q = q(rand_N)
    qq = _Primed(q)
    dqq = _DerivativeSymbol(qq)
    p = p(rand_N)
    pp = _Primed(p)
    dpp = _DerivativeSymbol(pp)
    a = alpha(rand_N)
    ad = alphaD(rand_N)
        
    def test_bopp_shift(self):
        q_bopp_right =  Bopp(q(), left=False)
        q_bopp_left = Bopp(q(), left=True)
        p_bopp_right = Bopp(p(), left=False)
        p_bopp_left = Bopp(p(), left=True)
        ddq = _DerivativeSymbol(_Primed(q()))
        ddp = _DerivativeSymbol(_Primed(p()))
        
        for bopped, check in zip([q_bopp_right, 
                                  q_bopp_left, 
                                  p_bopp_right, 
                                  p_bopp_left],
                                 [q() + sp.I*hbar/2*ddp,
                                  q() - sp.I*hbar/2*ddp,
                                  p() - sp.I*hbar/2*ddq,
                                  p() + sp.I*hbar/2*ddq]):
            assert (bopped - check).expand() == 0
            assert not(bopped.is_commutative)
            
    def test_fido(self):
        
        def FIDO(x):
            return _first_index_and_diff_order(x)
        
        try:
            FIDO(self.x+2+self.dqq)
            raise ValueError("Input should be invalid.")
        except:
            pass
        
        assert FIDO(1*self.x*self.q*self.pp) is None
        assert FIDO(self.qq**5) is None
       
        assert FIDO(self.dqq) == (0, self.qq, 1)
        assert FIDO(self.dpp) == (0, self.pp, 1)
        
        assert FIDO(self.dqq**self.rand_N) == (0, self.qq, self.rand_N)
        assert FIDO(self.dpp**self.rand_N) == (0, self.pp, self.rand_N)
    
        assert FIDO(self.qq**5*self.p*self.dqq*self.pp) == (2, self.qq, 1)
        assert FIDO(self.dpp*self.dqq) == (0, self.pp, 1)
        
        random_symbols = [sp.Symbol(r"TEST-{%s}" % n, commutative=False) for n in range(100)]
        random_symbols[self.rand_N] = self.dqq
        assert FIDO(sp.Mul(*random_symbols)) == (self.rand_N, self.qq, 1)
        
    def test_replace_diff(self):
        WW = _Primed(W())
        
        assert _replace_diff(sp.Integer(1)) == 1
        assert _replace_diff(self.x) == self.x
        assert _replace_diff(self.dqq) == sp.Derivative(1, self.qq, evaluate=False)
        
        assert _replace_diff(self.dqq*WW) == sp.Derivative(WW, self.qq)
        assert _replace_diff(self.dpp*WW) == sp.Derivative(WW, self.pp)
        
        assert (_replace_diff(self.dqq**2*self.dpp*WW) 
                == sp.Derivative(sp.Derivative(WW, self.pp), 
                                 self.qq, 2, evaluate=False))
        
        assert (_replace_diff(self.dqq*self.qq*self.pp*WW) 
                == sp.Derivative(self.qq*self.pp*WW, self.qq, evaluate=False))
        
    def test_star_base(self):
        def must_raise_error(bad_A, bad_B):
            try:
                _star_base(bad_A, bad_B)
                raise ValueError("Input should be invalid.")
            except:
                pass    
        for bad_A, bad_B in [[sp.sqrt(self.q), sp.sqrt(self.p)],
                             [sp.Function("foo_A")(self.q, self.p), W()],
                             [self.q**0.2, self.p**1.0000]]:
            must_raise_error(bad_A, bad_B)
        
        q0, p0, a0, ad0 = self.q, self.p, self.a, self.ad
        q1, p1, a1, ad1 = q(self.rand_N+1), p(self.rand_N+1), alpha(self.rand_N+1), alphaD(self.rand_N+1)
        for A, B, out in [[q0, q0, q0**2],
                          [p0, p0, p0**2],
                          [q0, p0, p0*q0 + sp.I*hbar/2],
                          [p0, q0, p0*q0 - sp.I*hbar/2],
                          [a0, ad0, (q0**2+p0**2+hbar)/(2*hbar)],
                          [ad0, a0, (q0**2+p0**2-hbar)/(2*hbar)],
                          [q0, p1, q0*p1],
                          [p0, q1, p0*q1],
                          [a0, ad1, a0*ad1],
                          [ad0, a1, ad0*a1]]:
            
            assert (_star_base(A, B) - out).expand() == 0
        
    def test_star(self):
        assert Star() == 1
        assert Star(self.q) == self.q
        for n in range(2, 5):
            assert Star(*[self.q]*n) == self.q**n