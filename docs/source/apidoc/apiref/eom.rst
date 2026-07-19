Equations of Motion
===================

This package is written to more easily obtain the phase-space representation of equations of motions originally
formulated in the Hilbert space.

The Linblad master equation
---------------------------

A nice tutorial can be found at [Manzano2020]_. The Lindblad master equation is a widely-used model for 
open quantum systems, thanks to its simplicity being somewhat an extension to the Liouville-von Neumann 
equation for Hamiltonian quantum systems. Its general form is given by 

.. math::

    \frac{\mathrm{d}\rho}{\mathrm{d}t} = -i\hbar \left[\hat{H},\rho\right] + \sum_{j,k} \gamma_{jk} \mathcal{D}\left(\hat{F}_j,\hat{F}_k\right)\left[\rho\right],

where 

.. math::

    \mathcal{D}\left(\hat{F}_j,\hat{F}_k\right)\left[\rho\right]
    =
    \hat{F}_j\rho\hat{F}_k - \frac{1}{2} \left(\hat{F}_k^\dagger \hat{F}_j\rho + \rho \hat{F}_k^\dagger \hat{F}_j\right)

is the Lindblad dissipator. The rate coefficients :math:`\left\{\gamma_{jk}\right\}` must form a positive semidefinite matrix. In the 
case that it is diagonal and hence real-valued, it is customary to use the jump operators :math:`\sqrt{\gamma_{jj}}\hat{F}_j=\hat{L}_j`, thereby abosrbing
the rate coefficients.

.. autoclass:: symqups.eom.LindbladMasterEquation
    :members:
    :special-members: __new__

.. autoclass:: symqups.eom.LME