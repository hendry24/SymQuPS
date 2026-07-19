Object Grouping
===============

Inside SymQuPS' machinery, different algebraic manipulations are done depending on what the input expression contains. Since 
SymPy's object grouping is mathematically generic, it cannot capture the algebraic features of the Cahill-Glauber formalism.
As such, the package needs to implement its own object grouping. The package's object grouping is meant for the package itself,
so it is considered one of the *internal* functionalities.

A user may access the object grouping classes by writing the following import statement:

.. code-block:: python

    from symqups._internal.grouping import ...

Here are the package's object grouping classes:

.. automodule:: symqups._internal.grouping
    :members:
    :member-order: bysource