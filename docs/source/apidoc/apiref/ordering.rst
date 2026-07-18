Operator Ordering
=================

.. When doing canonical quantization, we are immediately faced with the dilemma of operator ordering: to promote 
.. classical observables such as :math:`\alpha^*\alpha`, how should we order :math:`\hat{a}` and :math:`\hat{a}^\dagger`? 
.. We can do :math:`\hat{a}\hat{a}^\dagger` or :math:`\hat{a}^\dagget\hat{a}` or even :math:`\pi \hat{a}\hat{a}^\dagger + (1-\pi)\hat{a}^\dagget\hat{a}`;
.. all are equally valid. 

Given a number :math:`m` of :math:`\hat{a}^\dagger` and a number :math:`n` of :math:`\hat{a}`, the Cahill-Glauber framework
admits a continuous family of operator orderings, parameterized by :math:`s` (see :py:data:`symqups.s`). Other than the 
"big three" orderings (normal, symmetric/Weyl, anti-normal) corresponding to the values :math:`s=1,0,-1`, respectively, the generic 
s-ordered monomial :math:`\left\{\left(\hat{a}^\dagger\right)^m\hat{a}^n\right\}_s` do not have explicit forms (as opposed to, say,
:math:`\left\{\left(\hat{a}^\dagger\right)^m\hat{a}^n\right\}_1=\left(\hat{a}^\dagger\right)^m\hat{a}^n`). One ordering is related to the 
other via the equation [Cahill1969a]_ 

.. math::

    \left\{\left(\hat{a}^\dagger\right)^m\hat{a}\right\}_s
    =
    \sum_{k=0}^{\min(m,n)} k! \binom{m}{k} \binom{n}{k} \left(\frac{t-s}{2}\right)^k
    \left\{\left(\hat{a}^\dagger\right)^{m-k}\hat{a}^{n-k}\right\}_t 

.. automodule:: symqups.ordering
    :members:
    :special-members: __new__