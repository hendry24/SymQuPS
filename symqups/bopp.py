def _hatted_star_dBopp_monomial_A_times_B(A : sp.Expr, B : sp.Expr) -> sp.Expr:
    
    s = CahillGlauberS.val
    
    if A.is_Mul:
        args = A.args
    else:
        args = [A]
        
    non_psvo = sp.Integer(1)
    out = B
    
    # NOTE: Since operator products are not commutative, we need
    # to work from the rightmost argument or A.
    
    # NOTE: The hatted star product is commutative so we only have one
    # variant of dBopp, i.e. we can always dBopp toward the right direction.
    
    for arg in reversed(args):
        if arg.has(createOp):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
                
            sub = list(arg.atoms(createOp))[0].sub
            a, ad = annihilateOp(sub), createOp(sub)
            
            for _ in range(n):
                out = _commutativeOp(ad)*out + sp.Rational(1,2)*(s-1) * Commutator(ad, out)
                
        elif arg.has(annihilateOp):
            
            if arg.is_Pow:
                n = arg.args[1]
            else:
                n = 1
                
            sub = list(arg.atoms(annihilateOp))[0].sub
            a, ad = annihilateOp(sub), createOp(sub)
            
            for _ in range(n):
                out = _commutativeOp(a)*out - sp.Rational(1,2)*(s+1) * Commutator(a, out)
        else:
            non_psvo *= arg
            
    return _detilde(non_psvo*out.doit().expand())
