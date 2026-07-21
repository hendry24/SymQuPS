Constants
=========

There are four variables representing the constants in SymQuPS. Except for :py:data:`symqups.pi`, the value of 
these constants can be retrieved using ``const.val`` and set using ``const.val = myValue``. The default value can be retrieved
using ``const.default_value``, so a constant can be set to its default value using ``const.val = const.default_value``.

.. warning::

    The constant objects themselves cannot be used algebraically. Be sure to retrieve their values for algebraic manipulations.

.. py:data:: symqups.s

    The Cahill-Glauber ordering parameter, which determines the value of :math:`s` used in all functionalities 
    of the package, except those with an explicit ordering parameter as arguments like :class:`symqups.ordering.sOrdering`.
    Can take on any real value between :math:`-1` and :math:`1`, inclusive. Defaults to the symbol :math:`s`.

.. py:data:: symqups.hbar

    The reduced Planck's constant :math:`\hbar`. Defaults to the symbol :math:`\hbar` and can be set to any positive real value, 
    though setting :math:`\hbar=1` is probably the only worthwhile choice.

.. py:data:: symqups.zeta

    The scaling parameter for the complexified phase space. It is the value :math:`\zeta` such that 
    
    .. math::
    
        \alpha = \frac{\zeta q + i \zeta^{-1} p}{\sqrt{2\hbar}},
    
    Defaults to the symbol :math:`\zeta` and can be set to any positive real value, such as :math:`\sqrt{m\omega}` for 
    a simple harmonic oscillator.