class _CommutatorSymbol(Base, HilbertSpaceObject, UnDBoppable):
    def _get_symbol_name_and_assumptions(cls, left):
        return r"\left[%s, \cdot\right]" % (sp.latex(left)), {"commutative" : False}

    def __new__(cls, left : sp.Expr):
        return super().__new__(cls, left)
    
    @property
    def left(self):
        return self._custom_args[0]
    
def _symb2comm(expr : sp.Expr) -> sp.Expr:
    # We put this here insteaad of 'manipulations' since we only use this here. 
    
    def treat_add(A : sp.Add) -> sp.Expr:
        return sp.Add(*mp_helper(A.args, _symb2comm))
    
    def fico(A : sp.Expr) -> None | Tuple[int, Operator, int|sp.Integer]: # first index and commutator order. 
        def treat_mul(AA : sp.Mul) -> tuple:
            for idx, arg in enumerate(AA.args):
                if isinstance(arg, _CommutatorSymbol):
                    return idx, arg.left, 1
                if arg.has(_CommutatorSymbol):
                    return idx, arg.args[0].left, arg.args[1]
        
        return operation_routine(A,
                                "_symb2comm.fico",
                                [sp.Add],
                                [],
                                {_CommutatorSymbol : None},
                                {_CommutatorSymbol : lambda A: (0, A.left, 1),
                                sp.Pow : lambda A: (0, A.args[0].left, A.args[1]),
                                sp.Mul : treat_mul}
                                )
    def replace_comm(A : sp.Expr) -> sp.Expr:
        fico_res = fico(A)
        
        if fico_res:
            cut_idx, comm_left, comm_order = fico_res
            prefactor = A.args[:cut_idx]
            A_leftover = _CommutatorSymbol(comm_left)**(comm_order-1) * sp.Mul(*A.args[cut_idx+1:])
            # NOTE: This is where it differs from sp.Derivative. We can't just shortcut and do multiple 
            # commutator brackets. 
            return sp.Mul(*prefactor,
                            spq.Commutator(comm_left, replace_comm(A_leftover)))
            
        return A
    
    return operation_routine(expr,
                            "_symb2der",
                            [],
                            [],
                            {_CommutatorSymbol : expr},
                            {sp.Add : treat_add,
                            (sp.Mul, sp.Pow, _CommutatorSymbol) : replace_comm}
                            )

def _eval_dBopp(expr: sp.Expr) -> sp.Expr:
    return _subs_template(expr, dBopp_dict, (AndClass(Operator, qpType),
                                             AndClass(Operator, alphaType)))

class dBopp(sp.Expr, UnDBoppable):
    def __new__(cls, expr : sp.Expr) -> sp.Expr:
        expr = sp.sympify(expr)
        
        if expr.has(UnDBoppable):
            return super().__new__(cls, expr)
        
        res = _eval_dBopp(expr)
        
        return res
    
    def _latex(self, printer) -> str:
        return r"\widetilde{\mathrm{Bopp}}\left[{%s}\right]" % sp.latex(self.args[0])
    
def _dual_star_base(A : sp.Expr, B : sp.Expr) -> sp.Expr:
    
    if any(not(obj.has(Operator)) for obj in [A,B]):
        return A*B
    
    cannot_dBopp_A = A.has(UnDBoppable) or not(A.is_polynomial(Operator))
    cannot_dBopp_B = B.has(UnDBoppable) or not(A.is_polynomial(Operator))
    
    if cannot_dBopp_A and cannot_dBopp_B:
        raise _CannotBoppFlag()
    
    if cannot_dBopp_A:
        X = dBopp(B)*A
    else:
        X = dBopp(A)*B
        
    return _symb2comm(sp.expand(X.doit()))
    
class dStar(sp.Expr, UnDBoppable, Defined):
    @staticmethod
    def _definition():
        out = r"f\left(\hat{\bm{a}},\hat{\bm{a}}^\dagger\right)"
        out += r"\mathbin{\widetilde{\star}_s}"
        out += r"g\left(\hat{\bm{a}},\hat{\bm{a}}^\dagger\right)"
        out += r"= f\left(\hat{\bm{a}}-\frac{1+s}{2}\left[\hat{\bm{a}},\cdot\right],"
        out += r"\hat{\bm{a}}^\dagger - \frac{1-s}{2}\left[\hat{\bm{a}}^\dagger, \cdot\right]\right)"
        out += r"g\left(\hat{\bm{a}},\hat{\bm{a}}^\dagger\right)"
        return sp.Symbol(out)
    definition = _definition()
    
    def __new__(cls, *args) -> sp.Expr:
        if not(args):
            return sp.Integer(1)
        
        # NOTE: dStar of two polynomials equals the s-ordering of their product,
        # since the iCGTransform of a product of two polynomials is that product,
        # converted into operators, and s-ordered. This helps shortcut the evaluation.
        
        undboppable_args = []
        
        out = sp.Integer(1)
        for k,arg in enumerate(args):
            try:
                if arg.has(qpType):
                    arg = qp2alpha(arg)
                out = _dual_star_base(out, arg)
            except _CannotBoppFlag:
                if out != 1:
                    undboppable_args.append(out)
                out = arg
                if k == (len(args)-1):
                    undboppable_args.append(arg)
                    
        if undboppable_args:
            return super().__new__(cls, *undboppable_args)
    
        return out
    
    def _latex(self, printer):
        out = r"\left({%s}\right)" % sp.latex(self.args[0])
        for arg in self.args[1:]:
            out += r"\widetilde{\star}_s \left({%s}\right)" % sp.latex(arg)
        return out