The Bopp Actors
===============

The "Bopp actors" is a term we use to refer to the fact that the Bopp shift evaluations of star products and
hatted star products summarized in :doc:`star` can be seen as differential operators and adjoint
superoperators, respectively. While the former is not so useful, the latter has a nice commutative property that
allows for significant speed-up for :class:`symqups.cg.iCGTransform` for non-polynomial inputs (e.g., when 
inverse-transforming an equation of motion). Both are probably not interesting to a typical quantum theorist, 
but we think it is nice to provide these functionalities in the pacakge for the user, and have thus implemented them.

.. automodule:: symqups.bopp
    :members: