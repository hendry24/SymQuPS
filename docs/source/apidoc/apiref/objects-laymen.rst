Objects in SymQuPS (For Typical Users)
======================================

This page lists all objects within the package that a typical user may use in the algebra system. 

Hilbert-space Operators
-----------------------

.. autoclass:: symqups.objects.operators.qOp
    :members: has_sub, sub, dagger
    :special-members: __new__

.. autoclass:: symqups.objects.operators.pOp
    :members: has_sub, sub, dagger
    :special-members: __new__

.. autoclass:: symqups.objects.operators.annihilateOp
    :members: has_sub, sub, dagger
    :special-members: __new__

.. autoclass:: symqups.objects.operators.createOp
    :members: has_sub, sub, dagger
    :special-members: __new__

.. py:data:: symqups.rho

    The time-dependent density matrix object.

Phase-space Scalars
-------------------

.. autoclass:: symqups.objects.scalars.q
    :members: has_sub, sub, conjugate
    :special-members: __new__

.. autoclass:: symqups.objects.scalars.p
    :members: has_sub, sub, conjugate
    :special-members: __new__

.. autoclass:: symqups.objects.scalars.alpha
    :members: has_sub, sub, conjugate
    :special-members: __new__

.. autoclass:: symqups.objects.scalars.alphaD
    :members: has_sub, sub, conjugate
    :special-members: __new__

.. py:data:: symqups.W

    The state function object, which is the phase space representation of the density matrix
    within the Cahill-Glauber formalism. For :math:`s=-1`, the state function is :math:`(\pi P)`, where
    :math:`P` is the Glauber-Sudarshan :math:`P` representation.
    For :math:`s=0`, the state function is the Wigner function :math:`W`. For :math:`s=1`, the state function
    is the Husimi :math:`Q` function. The arguments of the state function is automatically updated by the package
    every time a phase space variable or its corresponding operator with a new subscript is added, so a typical 
    user need not treat this object as anything beyond an algebraic object.