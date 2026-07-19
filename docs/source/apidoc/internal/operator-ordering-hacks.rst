Operator Ordering Hacks
=======================

The core functionalities of SymPy has its own rule for ordering noncommutative objects, which is not exactly what we envision
to nicely order ``Operator``s in this package. As such, we have added a simple *patch* to the ``flatten`` method of the ``Mul``
class, where operator ordering is done such that ``Operator``s sharing the same ``sub`` are printed close together. The ordering
of the operator "groups" themselves are based on the order in which the ``sub``s are added to the ``sub_cache`` (see :doc:`caching`).

This hack indeed tampers with SymPy's core functionalities, but everything should work fine provided that a user only uses SymQuPS alongside
the core SymPy implementation (meaning no extra functionalities like ``sympy.physics``). 