Subscript Caching
=================

Every time an object with ``has_sub=True`` is created (see :doc:`../apiref/objects-laymen` and :doc:`../apiref/objects-powerusers`),
the package checks whether its ``sub`` already exists. If not, then that subscript is added to the ``sub_cache`` and the
package refreshes relevant data to reflect the addition of "new subsystems" in the session. This includes:

- Updating the list of arguments for `symqups.W` so that it can be differentiated with respect to the newly-created phase-space variables, thereby allowing the state function to adapt to the expanded system description.

- Updating the operator-to-scalar and scalar-to-operator substitution dictionaries, which are used to promote phase-space scalars to Hilbert space operators and vice-versa.

- Updating the qp-to-alpha and alpha-to-qp substitution dictionaries, which are used to convert between the :math:`(q,p)` canonical phase space and the :math:`(\alpha, \alpha^*)` complexified phase space, and similarly for their operator counterparts.