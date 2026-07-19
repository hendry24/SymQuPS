The Bopp Actors
===============

The "Bopp actors" is a term we use to refer to the fact that the Bopp shift evaluations of star products and
hatted star products summarized in :doc:`star` can be seen as differential operators and adjoint
superoperators, respectively. While the former is not so useful, the latter has a nice commutative property that
allows for significant speed-up for :class:`symqups.cg.iCGTransform` for non-polynomial inputs (e.g., when 
inverse-transforming an equation of motion). Both are probably not interesting to a typical quantum theorist, 
but we think it is nice to provide these functionalities in the package for the user, and have thus implemented them.
These properties are discussed by [Lim2025]_.

The phase-space Bopp operators
------------------------------

Given real-analytic :math:`\hat{F}=\hat{F}\left(\hat{a},\hat{a}^\dagger \right)`, we have the property that (see :doc:`cg-quantization` and :doc:`star`)

.. math::

    \begin{aligned}
    \mathcal{W}_s \left(\hat{F}\hat{G}\right) 
    &= 
    \hat{F}\left(\overset{\rightarrow}{\mathcal{B}}_\alpha, \overset{\rightarrow}{\mathcal{B}}_{\alpha^*}\right) \mathcal{W}_s \left(\hat{G}\right)
    \\
    &= 
    \mathcal{W}_s \left(\hat{F}\right)\hat{G}\left(\overset{\leftarrow}{\mathcal{B}}_\alpha, \overset{\leftarrow}{\mathcal{B}}_{\alpha^*}\right),
    \end{aligned}

where 

.. math::
    \begin{aligned}
        \overset{\rightarrow}{\mathcal{B}}_{\alpha} &= \alpha + \frac{s+1}{2} \overset{\rightarrow}{\partial}_{\alpha^*},
        \\
        \overset{\rightarrow}{\mathcal{B}}_{\alpha^*} &= \alpha^* + \frac{s-1}{2} \overset{\rightarrow}{\partial}_{\alpha},
        \\
        \overset{\leftarrow}{\mathcal{B}}_{\alpha} &= \alpha + \frac{s-1}{2} \overset{\leftarrow}{\partial}_{\alpha^*},
        \\
        \overset{\leftarrow}{\mathcal{B}}_{\alpha^*} &= \alpha^* + \frac{s+1}{2} \overset{\leftarrow}{\partial}_{\alpha},
    \end{aligned}

are the phase-space Bopp operators (PSBOs). Details on the directional derivatives can be found in :doc:`star`. Like the operators they replace,
the PSBOs do not commute with each other. 

.. autoclass:: symqups.bopp.PhaseSpaceBoppOperator
    :members: base, left, act
    :special-members: __new__

.. autoclass:: symqups.bopp.PSBO
    :members:

The Hilbert-space Bopp superoperators
-------------------------------------

Given real-analytic :math:`f=f\left(\alpha,\alpha^* \right)`, we have the property that (see :doc:`cg-quantization` and :doc:`star`)

.. math::

    \begin{aligned}
    \mathcal{W}_s^{-1} \left(fg\right) 
    &= 
    f\left(\overset{\rightarrow}{\mathcal{B}}_{\hat{a}}, \overset{\rightarrow}{\mathcal{B}}_{\hat{a}^\dagger}\right) \mathcal{W}_s^{-1} \left(g\right)
    \\
    &= 
    \mathcal{W}_s^{-1} \left(f\right) g \left(\overset{\leftarrow}{\mathcal{B}}_{\hat{a}}, \overset{\leftarrow}{\mathcal{B}}_{\hat{a}^\dagger}\right)
    \end{aligned}

where 

.. math::
    \begin{aligned}
        \overset{\rightarrow}{\mathcal{B}}_{\hat{a}} &= \hat{a} - \frac{s+1}{2} \overset{\rightarrow}{\partial}_{\hat{a}^\dagger},
        \\
        \overset{\rightarrow}{\mathcal{B}}_{\hat{a}^\dagger} &= \hat{a}^\dagger - \frac{s-1}{2} \overset{\rightarrow}{\partial}_{\hat{a}},
        \\
        \overset{\leftarrow}{\mathcal{B}}_{\hat{a}} &= \hat{a} - \frac{s-1}{2} \overset{\leftarrow}{\partial}_{\hat{a}^\dagger},
        \\
        \overset{\leftarrow}{\mathcal{B}}_{\hat{a}^\dagger} &= \hat{a}^\dagger - \frac{s+1}{2} \overset{\leftarrow}{\partial}_{\hat{a}}, 
    \end{aligned}

are the Hilbert-space Bopp superoperators (HSBSs). Details on the directional derivatives can be found in :doc:`star`. 
Like the scalars they replace, the HSBSs commute with each other, which comes in handy to avoid the need to evaluate the 
hatted star products when doing the inverse Cahill-Glauber transform. As such, they are implemented in :class:`symqups.cg.iCGTransform`,
though its implemented object class is not used to make calculations more efficient by hard-coding the HSBSs directly. 

.. autoclass:: symqups.bopp.HilbertSpaceBoppSuperoperator
    :members: base, left, act
    :special-members: __new__

.. autoclass:: symqups.bopp.HSBS
    :members: