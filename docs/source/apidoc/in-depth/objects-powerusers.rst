Objects in SymQuPS (For Power Users)
====================================

This page lists all object classes of the package that is meant for power users. A typical user
should not need to deal with these functionalities and may refer to :doc:`../basic/objects-laymen`.

..  autoclass:: symqups.objects.base.Base

..  autoclass:: symqups.objects.operators.Operator
    :members: has_sub, sub, dagger
    :special-members: __new__

.. autoclass:: symqups.objects.operators.HermitianOp
    :members: has_sub

.. autoclass:: symqups.objects.operators.densityOp
    :members: has_sub

.. autoclass:: symqups.objects.operators.TimeDependentOp
    :special-members: __new__

..  autoclass:: symqups.objects.scalars.Scalar
    :members: has_sub, sub, conjugate
    :special-members: __new__
    
.. autoclass:: symqups.objects.scalars.t
    :members: has_sub

.. autoclass:: symqups.objects.scalars.StateFunction