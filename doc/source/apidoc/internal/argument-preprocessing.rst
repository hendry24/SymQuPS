Argument Preprocessing
======================

All top-level functionalities of the package is decorated with the ``preprocess_func`` or ``preprocess_class`` decorator,
which does two things:

-   It "sympifies" all arguments to be SymPy-compatible.

-   It calls ``.doit`` followed by ``.expand`` method to cast input expressions into a form the package works the best with.

Typically, this casts relevant input expressions into a polynomial in the phase-space scalars or their operator counterparts
in their standard form, which is an ``sympy.Add`` objects containing ``sympy.Mul`` containing no more ``sympy.Add`` objects.