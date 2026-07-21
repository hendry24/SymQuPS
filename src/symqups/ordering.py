import sympy as sp
from itertools import permutations, product
import functools

from ._internal.grouping import HilbertSpaceObject, CannotBoppShift, PhaseSpaceVariableOperator
from ._internal.cache import sub_cache
from ._internal.basic_routines import (operation_routine, 
                                       default_treat_add,
                                       EmptyPlaceholder)
from ._internal.math import has_universal_oper
from ._internal.preprocessing import preprocess_func

from .objects.operators import Operator, annihilateOp, createOp

from .manipulations import qp2alpha, explicit_sOrdering

###

class sOrdering(sp.Expr, HilbertSpaceObject, CannotBoppShift):
    """
    The s-ordering bracket.
    """
    
    is_commutative = False
    
    @preprocess_func    
    def __new__(cls, expr : sp.Expr, s : float = None, 
                _fast_constructor : tuple = None) -> sp.Expr:
        
        """
        Construct an s-ordering bracket.
        
        Parameters
        ----------
        
        expr : sympy.Expr
            The expression to be enclosed. If no ``Operator``s are present, then the expression is
            returned as is. If ``Operator``s with ``has_sub=False`` like ``rho`` is contained, then 
            an error is raised.
            
        s : float, optional
            Ordering parameter for the bracket. This input does not affect the package variable ``s``.
            If a number is passed, the package raises a warning if it is not a real number between
            -1 and 1, inclusive. Using any ``sympy.Symbol`` is allowed. By default, the current value
            of ``symqups.s`` is used.
            
        _fast_constructor : tuple, default: None
            A tuple `(poly_dict, nonpoly_args)` for fast construction of the bracket. Used internally 
            by the package, but the user is welcome to try.
        
        """
        
        from . import s as CahillGlauberS
        
        if s is None:
            s = CahillGlauberS.val
        else:
            # We need to trigger warning in case a bad `s` is input by trying to
            # set the value of the package's `s`. After that, we return to the 
            # current set value. 
            current_s_value = CahillGlauberS.val
            CahillGlauberS.val = s # This will trigger a warning for bad `s`.
            CahillGlauberS.val = current_s_value
            
        def has_ordering_ambiguity(A : sp.Expr) -> bool:
            if any(A.has(annihilateOp(sub)) and A.has(createOp(sub))
                   for sub in sub_cache):
                return True
            return False
            
        def make(poly_dict : dict,
                 nonpoly_args: tuple) -> sOrdering:
            A = sp.Mul(*[b**e for sub,pow_lst in poly_dict.items() 
                         for b,e in zip([createOp(sub), annihilateOp(sub)],
                                        pow_lst)],
                        *nonpoly_args)
            if not(has_ordering_ambiguity(A)):
                return A
            obj = super(cls, cls).__new__(cls, A, s, EmptyPlaceholder())
            obj._poly_dict = poly_dict
            obj._nonpoly_args = nonpoly_args
            return obj
        
        if not(isinstance(_fast_constructor, (type(None), EmptyPlaceholder))):
            poly_dict, nonpoly_args = _fast_constructor
            return make(poly_dict, nonpoly_args)
        
        ###
        
        # We assume that the input does not contain any universally-noncommuting
        # operators like 'densityOp'.
        if has_universal_oper(expr):
            msg = "No universal operators should be put into s-ordering. "
            msg += "Input may contain 'densityOp' which never goes in "
            msg += "the ordering braces."
            raise ValueError(msg)
        
        def treat_add(A : sp.Expr) -> sp.Expr:
            return default_treat_add(A.args, functools.partial(sOrdering,
                                                               s=s))
                    
        def treat_pow(A : sp.Expr) -> sp.Expr:
            if A.is_polynomial(Operator):
                return A
            return make({}, [A])
            
        def treat_function(A : sp.Function) -> sp.Expr:
            if has_ordering_ambiguity(A):
                return make({}, [A])
            return A
        
        def treat_mul(A : sp.Expr) -> sp.Expr:
            if not(has_ordering_ambiguity(A)):
                return A
            
            # We don't care about operator ordering inside
            # the braces, so might as well return it pretty.
            coefs = []
            poly = {sub : [
                0, # number of polynomial ad_sub
                0, # number of polynomial a_sub
            ] for sub in sub_cache}
            nonpoly = [] 
            # We shove all nonpolynomial to the right of the polynomial part
            # since one nonpoly factor
            # may contain multiple subs. Separating "separable" factors
            # into their own ordering braces would be unnecessarily 
            # expensive, so we don't do that.
            #
            for arg in A.args:
                arg : sp.Expr
                if arg.has(PhaseSpaceVariableOperator):
                    if arg.is_polynomial(PhaseSpaceVariableOperator):
                        b, e = arg.as_base_exp()
                        if isinstance(b, createOp):
                            poly[b.sub][0] += e
                        elif isinstance(b, annihilateOp):
                            poly[b.sub][1] += e
                        else:
                            raise ValueError("Invalid value.")
                    else:
                        nonpoly.append(arg)
                else:
                    coefs.append(arg)
            
            in_braces = make(poly, nonpoly)
            
            return sp.Mul(*coefs, in_braces)
        
        expr = qp2alpha(expr)
        return operation_routine(sp.expand(sp.sympify(expr)),
                                  sOrdering,
                                  [],
                                  [],
                                  {Operator : expr},
                                  {(Operator, sOrdering) : expr,
                                   sp.Pow : treat_pow,
                                   sp.Function : treat_function,
                                   sp.Mul : treat_mul,
                                   sp.Add : treat_add}
                                  )
    
    @property
    def content(self):
        """
        Content of the bracket. Generally not equivalent to the input expression.
        """
        
        return self.args[0]
    
    @property
    def s_val(self):
        """
        Ordering paramter of the bracket.
        """
        
        return self.args[1]
    
    @property
    def poly_dict(self) -> dict:
        """
        Polynomial dict in ``annihilateOp`` and ``createOp`` with the format
        `{sub : (power of createOp(sub), power of annihilateOp(sub))}`
        """
        
        return self._poly_dict
    
    @property
    def nonpoly_args(self) -> list:
        """
        List of non-polynomial factors of the content. 
        """
        
        return self._nonpoly_args
    
    @property
    def contains_poly(self) -> bool:
        """
        Whether the content is a polynomial.
        """
        
        return not(self.nonpoly_args)
            
    def _latex(self, printer) -> str:
        return r"\left\{ %s \right\}_{s=%s}" % (sp.latex(self.content), sp.latex(self.s_val))
        
    def explicit(self) -> sp.Expr:
        """
        Return the explicit form. Only applicable to s=-1,0,1 ordering brackets, otherwise
        the return ``self``. 
        """
        
        if not(self.contains_poly):
            return self
        
        match self.s_val:
            case 1:
                return self.content
            case 0:
                out_factors = []
                for sub in self.poly_dict.keys():
                    ad, m = createOp(sub), self.poly_dict[sub][0]
                    a, n = annihilateOp(sub), self.poly_dict[sub][1]
                    to_permutate = [ad]*m + [a]*n
                    out_single_sub_summands = []
                    for permutation in permutations(to_permutate, len(to_permutate)):
                        out_single_sub_summands.append(sp.Mul(*permutation))
                    if len(to_permutate) != 0:
                        out_factors.append(sp.cancel(sp.Add(*out_single_sub_summands) / sp.factorial(len(to_permutate))))
                return sp.Mul(*out_factors)
            case -1:
                return sp.Mul(*[b**e for sub,pow_lst in self.poly_dict.items() 
                         for b,e in zip([annihilateOp(sub),createOp(sub)],
                                        reversed(pow_lst))])
            case default:
                return self

    @preprocess_func
    def express(self, t = 1, explicit=True) -> sp.Expr:
        """
        Expand the expression in terms of t-ordered expressions.
        By default, `t=1` corresponds to normal-ordering. If `define`,
        then the expanded t-ordered expressiosn are defined when possible. 
        By default, express the object in terms of normal-ordered products.
        """
        
        if not(self.contains_poly):
            return self
        
        coef_lst = []
        poly_dict_val = [] 
        # Each entry is for one sub, and contains the terms of the
        # expanded series for that sub. Each term is not evaluated;
        # we have instead a list of factors. 
        # 
        # We multiply the series for all the subs, so this is the same as taking 
        # the Cartesian product of the terms. The Cartesian product of the items 
        # in poly_dict_val will be used to form the poly_dict to construct the 
        # sOrdering content for the given output term.
        #
        # Code may be less readable, but we want to go fast since this function 
        # is implemented in CGTransform.
        
        for sub in self.poly_dict.keys():
            m = self.poly_dict[sub][0]
            n = self.poly_dict[sub][1]
            terms_coef = []
            terms_poly_dict_val = []
            for k in range(min(m,n) + 1):
                terms_coef.append([
                    sp.factorial(k),
                    sp.binomial(m, k),
                    sp.binomial(n, k),
                    sp.Rational(1, 2**k),
                    (t-self.s_val)**k, 
                ])
                terms_poly_dict_val.append(
                    [m-k, n-k]
                )
            coef_lst.append(terms_coef)
            poly_dict_val.append(terms_poly_dict_val)
            
        def get_express_out_summand(inpt, t, explicit, sub_lst):
            coef_combo, poly_dict_val_combo = inpt
            coef = [c for c_lst in coef_combo for c in c_lst]
            poly_dict = {sub : mn for sub, mn in zip(sub_lst, poly_dict_val_combo)}
            sordered = sOrdering(1, s=t, _fast_constructor=[poly_dict, []])
            if explicit and isinstance(sordered, sOrdering):
                sordered = sordered.explicit()
            return sp.Mul(*coef, sordered)
         
        return sp.Add(*[get_express_out_summand(val, t, explicit, self.poly_dict.keys()) 
                        for val in list(zip(product(*coef_lst), product(*poly_dict_val))) 
                        ])

###

@preprocess_func
def normal_order(expr : sp.Expr) -> sp.Expr:
    """
    Normal-order ``expr``. The resulting expression is generally NOT equivalent.
    """
    return explicit_sOrdering(sOrdering(expr, s=1))

@preprocess_func
def antinormal_order(expr : sp.Expr) -> sp.Expr:
    """
    Anti-normal-order ``expr``. The resulting expression is generally NOT equivalent.
    """
    return explicit_sOrdering(sOrdering(expr, s=-1))

@preprocess_func
def Weyl_order(expr : sp.Expr) -> sp.Expr:
    """
    Weyl-order (or symmetric-order) ``expr``. The resulting expression is generally NOT equivalent.
    """
    return explicit_sOrdering(sOrdering(expr, s=0))