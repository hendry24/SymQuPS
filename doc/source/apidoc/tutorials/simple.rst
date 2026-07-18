Using ``symqups.simple`` for Unipartite Systems
===============================================

The best use case of the phase space representation is when our quantum system is **unipartite**, in which case the phase space 
is two dimensional. Starting with bipartite systems, we have four-dimensional phase space, which is not exactly fun to visualize.
As such, we expect most use cases to deal with unipartite systems, in which case the full power of SymQuPS is probbaly overkill.
To that end, we provide ``symqups.simple``, a simpler SymQuPS **without subscripts** where the objects have been created and 
assigned as package variables. This simplified package comes with SymQuPS and can be accessed using ``from symqups.simple import *``
(yes, we recommend pulling in everything there). A use that uses ``symqups.simple`` has probably no need to import from the main package.
Here are the variables and functionalities available on ``symqups.simple``.

Package variables
-----------------

Constants
~~~~~~~~~

.. py:data:: s

    The Cahill-Glauber ordering parameter :math:`s`. See :doc:`../apiref/constants`.

.. py:data:: hbar

    The reduced Planck's constant :math:`\hbar`. See :doc:`../apiref/constants`.

.. py:data:: zeta

    The scaling paramter :math:`\zeta` for :math:`\alpha`. See :doc:`../apiref/constants`.

.. py:data:: pi

    The transcendental number :math:`pi`. See :doc:`../apiref/constants`.

Scalars
~~~~~~~

.. py:data:: q

    The canonical position :math:`q`. An instance of :class:`symqups.objects.scalars.q`.

.. py:data:: p
    
    The canonical momentum :math:`p`. An instance of :class:`symqups.objects.scalars.p`.

.. py:data:: a

    The complexified phase-space variable :math:`\alpha`. An instance of :class:`symqups.objects.scalars.alpha`.

.. py:data:: ad

    The complexified phase-space variable :math:`\alpha^*`. An instance of :class:`symqups.objects.scalars.alphaD`.

.. py:data:: W

    The state function :math:`W_s`. See :doc:`../apiref/objects-laymen`.

Operators
~~~~~~~~~

.. py:data:: qOp
    
    The operator :math:`\hat{q}` corresponding to :math:`q`. An instance of :class:`symqups.objects.operators.qOp`.

.. py:data:: pOp
    
    The operator :math:`\hat{p}` corresponding to :math:`p`. An instance of :class:`symqups.objects.operators.pOp`.

.. py:data:: aOp
    
    The operator :math:`\hat{a}` corresponding to :math:`\alpha`. An instance of :class:`symqups.objects.operators.annihilateOp`.

.. py:data:: adOp
    
    The operator :math:`\hat{a}^\dagger` corresponding to :math:`\alpha^*`. An instance of :class:`symqups.objects.operators.createOp`.

.. py:data:: rho

    The time-dependent density operator or density matrix :math:`\rho`. See :doc:`../apiref/objects-laymen`.



Package functionalities
-----------------------

.. important::

    The user simply calls the function/class names without using the full module path as shown below. 

.. autosummary::
   :nosignatures:

   symqups.cg.CGTransform
   symqups.cg.iCGTransform
   
   symqups.eom.LindbladMasterEquation
   symqups.eom.LME
   
   symqups.manipulations.alpha2qp
   symqups.manipulations.qp2alpha
   symqups.manipulations.sc2op
   symqups.manipulations.op2sc
   symqups.manipulations.dagger
   symqups.manipulations.explicit_sOrdering
   symqups.manipulations.express_sOrdering
   symqups.manipulations.normal_ordered_equivalent
   symqups.manipulations.s_ordered_equivalent
   symqups.manipulations.Derivative
   symqups.manipulations.Commutator
   
   symqups.ordering.sOrdering
   symqups.ordering.normal_order
   symqups.ordering.Weyl_order
   symqups.ordering.antinormal_order
   
   symqups.star.Star
   symqups.star.HattedStar

   symqups.quantization.s_quantize
   symqups.quantization.normal_quantize
   symqups.quantization.Weyl_quantize
   symqups.quantization.antinormal_quantize
   
   symqups.utils.get_random_poly
   symqups.utils.collect_by_derivative