The Cahill-Glauber Correspondence
=================================

The Cahill-Glauber correspondence is a family of correspondence designed to form a correspondence between
an :math:`s`-ordered polynomial (see :doc:`ordering`) in :math:`(\hat{a},\hat{a}^\dagger)` and a scalar polynomial which has the
same number of :math:`\alpha` as :math:`\hat{a}`  and :math:`alpha^*` as :math:`\hat{a}^\dagger`, for each
term. That is, 

.. math::
    
    \mathcal{W}_s:\quad \left\{\left(\hat{a}^\dagger\right)^m \hat{a}^n\right\}_s \quad\mapsto\quad \left(\alpha^*\right)^m \alpha^n.

Meanwhile, the density matrix is mapped to the state function with a flipped :math:`s` sign:

.. math::
    
    \mathcal{W}_s:\quad \rho \quad\mapsto\quad W_{-s}\left(\alpha,\alpha^*\right).

The reason that we have :math:`W_{-s}` as opposed to :math:`W_s` is because the state function has a special property that 

.. math::

    \mathrm{tr}\left( \rho  \left\{\left(\hat{a}^\dagger\right)^m \hat{a}^n\right\}_t\right)
    =
    \int_{\mathbb{R}^2} \frac{\mathrm{d}^2\alpha}{\pi} \left(\alpha^*\right)^m \alpha^n W_{t}.

That is, the state function obtained by applying :math:`\mathcal{W}_s` has the above property for expectation values of quantities 
represented by :math:`(-s)`-ordered operators. 

The mapping :math:`\mathcal{W}_s` and its inverse :math:`\mathcal{W}_s^{-1}` are complex linear, e.g.,

.. math::

    \mathcal{W}_s\left(c_1 \hat{F}_1 + c_2 \hat{F}_2\right) 
    = 
    c_1 \mathcal{W}_s\left(\hat{F}_1\right) + c_2 \mathcal{W}_s\left(\hat{F}_2\right),
    \quad 
    c_{1,2} \in \mathbb{C}.

Furthermore, for products, the mappings has the properties that

.. math:: 

    \mathcal{W}_s \left(\hat{F}\hat{G}\right) = \mathcal{W}_s\left(\hat{F}\right) \mathbin{\star_s} \mathcal{W}_s\left(\hat{F}\right).

and 

.. math:: 

    \mathcal{W}_s^{-1} \left(fg\right) = \mathcal{W}_s^{-1}(f) \mathbin{\widehat{\star}_s} \mathcal{W}_s^{-1}(g).

The "forward" transform is implemented as ``CGTransform`` by the package, while the "inverse" is implemented as ``iCGTransform``. 
Though the correspondence is formally *analytic*, the above properties are what our implementation is based on. For a summary on the star products 
and the hatted star products, see :doc:`star`. 

Correspondence transforms
-------------------------

.. automodule:: symqups.cg
    :members: 
    :special-members: __new__
    
Quantization
------------

To a classical-mechanics description, the inverse Cahill-Glauber transform is the canonical quantization with 
ordering choice given by the value of :math:`s`. These functionalities are pretty much aliases to
:class:`iCGTransform`.

.. automodule:: symqups.quantization
    :members: