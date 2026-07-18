Star Products and their Hilbert space mirrors
=============================================

The star products
-----------------

The Cahill-Glauber transform of a product is equivalent to the star product between the CG transforms of its factors. 
For real-analytic functions like :math:`f=f\left(\alpha,\alpha^*\right)`, the star product in the CG correspondence framework
admits the differential form

.. math::

    f \mathbin{\star_s} g = f \exp\left( \frac{s+1}{2} \overset{\leftarrow}{\partial}_{\alpha} \overset{\rightarrow}{\partial}_{\alpha^*} + \frac{s-1}{2} \overset{\leftarrow}{\partial}_{\alpha^*} \overset{\rightarrow}{\partial}_{\alpha} \right) g.

.. important::

    The overset arrow means that a given differential operator operates only on the variables belonging to the function to its left/right.

The algebraic way to 
evaluate the star product is to use the Bopp shift evaluation. Based on the fact that :math:`\exp\left(c\partial_x\right)f(x)=f(x+c)` when :math:`c` does not
depend on :math:`x`, we can write

.. math::

    f \mathbin{\star_s} g = f \left(\frac{s+1}{2} \overset{\rightarrow}{\partial}_{\alpha^*}, \frac{s-1}{2}\overset{\rightarrow}{\partial}_{\alpha} \right)
    g(\alpha,\alpha^*).

The evaluation above Bopp-shifts the star product onto :math:`\alpha` and :math:`\alpha^*` belonging to :math:`f`, but we can also Bopp-shift the star product 
onto both variables to :math:`g` or do one-each for :math:`f` and :math:`g`. As an example, we have

.. math::

    \alpha \alpha^* \mathbin{\star}_s \alpha^* 
        &=
        \left[ \left(\alpha + \frac{s+1}{2}\overset{\rightarrow}{\partial}_{\alpha^*} \right)
        \left(\alpha^* + \frac{s-1}{2}\overset{\rightarrow}{\partial}_{\alpha} \right)
        \right] \alpha^*
    \\ 
        &=
        \left(\alpha^*\right)^2\alpha + \frac{s+1}{2} \alpha^*.

Note how the :math:`\overset{\rightarrow}{\partial}_{\alpha^*}` inside the first factor of the left-hand argument does not operate on the :math:`\alpha^*`
inside the second factor.  

.. autoclass:: symqups.star.Star
    :members:
    :special-members: __new__

The hatted star products 
------------------------

The inverse Cahill-Glauber transform of a product is equivalent to the hatted star product between the inverse CG transforms of 
the factors. Surprisingly, the hatted star products take a very elegant form in the CG correspondence framework. Given real-analytic 
functions like :math:`\hat{F}=\hat{F}\left(\hat{a},\hat{a}^\dagger\right)`, we have 

.. math::

    \hat{F} \mathbin{\widehat{\star}_s} \hat{G} = \hat{F} \exp\left( -\frac{s+1}{2} \overset{\leftarrow}{\partial}_{\hat{a}} \overset{\rightarrow}{\partial}_{\hat{a}^\dagger} - \frac{s-1}{2} \overset{\leftarrow}{\partial}_{\hat{a}^\dagger} \overset{\rightarrow}{\partial}_{\hat{a}} \right) \hat{G},

where the formal derivatives with respect to the ladder operators are defined as 

.. math::

    \partial_{\hat{a}}\hat{F} = \left[\hat{F}, \hat{a}^\dagger\right], \qquad \partial{\hat{a}^\dagger}\hat{F} = \left[\hat{a}, \hat{F}\right],

which works as if the operators were differentiation variables using the power rule for any monomial :math:`\hat{F}`.

Similarly, we can use Bopp shifts to algebraically evaluate the mappings. The directional derivatives are consistent with the commutator properties like 
:math:`\left[\hat{A},\hat{B}\hat{C}\right] = \left[\hat{A},\hat{B}\right]\hat{C} + \hat{B}\left[\hat{A},\hat{C}\right]` as long as we are mindful about the 
multiplication order. 

To implement the formal derivatives with respect to operators, SymQuPS has its own :class:`symqups.manipulations.Derivative` class which deals with the ladder operators 
passed as differentiation variables, turning them into :class:`symqups.manipulations.Commutator` before passing the resulting expression into ``sympy.Derivative`` alongside
the rest of the differentiation variables.

.. autoclass:: symqups.star.HattedStar
    :members:
    :special-members: __new__