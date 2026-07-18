.. SymQuPS documentation master file, created by
   sphinx-quickstart on Thu Jul 16 17:40:04 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introducing, SymQuPS!
=====================

SymQuPS is a computer algebra system based on SymPy that serves as a tool to algebraically go between the Hilbert space equipped with the canonical commutator
and the phase space equipped with the Poisson bracket structure, mainly to obtain the phase-space representation of quantum 
equations of motion to help quantum theorists with the cumbersome algebra. 

For descriptions that "start" in the Hilbert space, such as that of a quantum system involving the density matrix, transforming into the phase space gives
the **phase space representation** of that quantum system. On the flip side, for things that "start" in the phase
space, such as a classical observable function, the transformation into the Hilbert space is known as the 
**canonical quantization** of that classical observable, i.e., we find a quantum mechanical operator which represents
said observable.

It has been shown by [Groenewold1946]_ and [VanHove1951]_ that the transformation between the two spaces is not unique;
we can have multiple correspondences, each associated with a choice of operator ordering, e.g., :math:`qp\leftrightarrow \hat{q}\hat{p}`
and :math:`qp\leftrightarrow\hat{p}\hat{q}`. As such, there exists multiple, equally-valid, correspondences, each with its own
peculiarities. Choosing what to use, then, is a matter of asking what we want to see.

The algebra of quantum mechanical operators is *noncommutative*, while the algebra of classical observables is *commutative*.
As a consequence, the mapping of products is nontrivially related to the mappings of its factors. For Hilbert-to-phase-space 
transforms, the mapping of a product is given by the **star product** between the mappings of the factors, a prime example of
which is the Moyal product [Moyal1949]_. Meanwhile, the inverse mapping of a product is given by what we call the **hatted star
product** between the inverse mappings of the factors; the hatted star product associated with the Moyal product is Bracken's 
odot product [Bracken2003]_.

SymQuPS implements the Cahill-Glauber correspondence [Cahill1969a]_ [Cahill1969b]_, alongside the results
of [Soloviev2015]_ and [Lim2026]_ to implement the star products and their hatted counterparts. The correspondence is a continuous
family that interpolates between the "big three" of ordering choices: the antinormal, symmetric, and normal orderings. 
More details are available in the four references given in this paragraph, which are briefly summarized in our **API REFERENCE**. 

While motivated by the desire for convenience in obtaining the phase-space representation of equations of motion,
such as the Lindblad master equation [Manzano2020]_, SymQuPS has been developed as a calculator-esque environment that allows 
multiple ways to algebraically manipulate mathematical expressions involving the phase-space variables and their operaotr counterparts, 
with main focus on polynomials. Here we highlight some functionalities offered by the pacakge:

-  Create and algebraically manipulate expressions in the canonical phase-space variables :math:`(q,p)` and the corresponding 
   formal complexified phase-space variables :math:`(\alpha,\alpha^*)`, where 

   .. math::

      \alpha = \frac{\zeta q + i \zeta^{-1} p}{\sqrt{2\hbar}},

   as well as their operator counterparts, within the extent of core arithmetics (addition, multiplication, exponentiation) and 
   generic functions (:math:`\mathrm{exp},\mathrm{ln},` or undefined functions). See :doc:`apidoc/apiref/objects-laymen`.

-  Rewrite operator expressions in any ordering defined within the Cahill-Glauber framework. See :doc:`apidoc/apiref/ordering` and
   :doc:`apidoc/apiref/manipulations`. 

-  Compute an arbitrarily long chain of star products and hatted star products. See :doc:`apidoc/apiref/star`.

-  Compute the phase space representation of any expressions containing the quantum operators. In particular, compute the
   phase space representation of an arbitrary Lindblad master equation. See :doc:`apidoc/apiref/cg-quantization` and :doc:`apidoc/tutorials/eom`.

-  Compute the canonical quantization of any functions in the phase-space variables. See :doc:`apidoc/apiref/cg-quantization`.

-  The two transforms mentioned above are inverses of each other. As such, the package is able to compute the inverse transforms of
   arbitrary expressions.

.. tip::

   Though we say "arbitrary" and "any", the possible inputs for which the algebraic operations are *fully* evaluable is limited. To be more precise,
   the package is most effective if the mathematical description for your problem contains no more than one non-polynomials in the variables
   :math:`(q,p,\alpha,\alpha^*)` and their operator counterparts, including the density matrix. This is not a problem for the use case in 
   physics that we are envisioning: quantum equations of motions usually involves only polynomial generators (e.g., Hamiltonian and 
   Lindblad dissipators in a Lindblad master equation) alongside one density matrix on each term. 
   For the curious mathematicians, the package tries its best to evaluate the algebraic operations it is instructed to do, 
   up until it can no longer do so. 

**Cite us!** If you find the package useful in your research, consider citing our paper that showcases this package to the community.
You will contribute in boosting its credence within the circle of quantum theorists who finds it difficult doing all the algebra. Here 
is a BibTeX entry you can copy:

.. code-block:: bibtex

   @article{Lim_2026_SymQuPS,
     author  = {Lim, Hendry M. and 
                Dwiputra, Donny and
                Ukhtary, Shoufie M. and
                Nugraha, Ahmad R. T.},
     title   = {SymQuPS: Symbolic Quantum Phase Space Algebra in Python},
     journal = {arXiv},
     year    = {tbd},
     doi     = {tbd}
   }

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   apidoc/tutorials/install
   apidoc/tutorials/simple
   apidoc/tutorials/eom

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   apidoc/apiref/objects-laymen
   apidoc/apiref/objects-powerusers
   apidoc/apiref/manipulations
   apidoc/apiref/ordering
   apidoc/apiref/star
   apidoc/apiref/cg-quantization
   apidoc/apiref/bopp
   apidoc/apiref/other-utils

.. toctree::
   :maxdepth: 1
   :caption: Internal functionalities

   apidoc/internal/caching
   apidoc/internal/grouping
   apidoc/internal/operator-ordering-hacks
   apidoc/internal/argument-preprocessing
   apidoc/internal/internal-math-utilities

References
==========

.. [Groenewold1946] Groenewold, H. (1946). On the principles of elementary
   quantum mechanics. *Physica*, 12, 405. 
   https://doi.org/https://doi.org/10.1016/S0031-8914(46)80059-4

.. [VanHove1951] Van Hove, L. C. P. (1951). Sur certaines répresentations 
   unitaires d'un groupe infini de transformations. *Ph.D. thesis, Université
   libre de Bruxelles*. 

.. [Moyal1949] Moyal, J. E. (1949). Quantum mechanics as a statistical theory.
   *Mathematical Proceedings of the Cambridge Philosophical Society*, 45, 99.
   https://doi.org/10.1017/S0305004100000487 

.. [Bracken2003] Bracken, A. J. (2003). Quantum mechanics as an approximation
   to classical mechanics in Hilbert space. *Journal of Physics A: Mathematical 
   and General*, 36, L329. https://doi.org/10.1017/S0305004100000487

.. [Cahill1969a] Cahill, K. E. and Glauber, R. J. (1969).
   Density operators and quasiprobability distributions.
   *Physical Review*, 177, 1857.
   https://doi.org/10.1103/PhysRev.177.1857

.. [Cahill1969b] Cahill, K. E. and Glauber, R. J. (1969).
   Ordered expansions in boson amplitude operators.
   *Physical Review*, 177, 1882.
   https://doi.org/10.1103/PhysRev.177.1882

.. [Soloviev2015] Soloviev, M. A. (2015). Integral representations
   of the star product corresponding to the *s*-ordering of the 
   creation and annihilation operators. *Physica Scripta*, 90, 074008. 
   http://dx.doi.org/10.1088/0031-8949/90/7/074008

.. [Lim2026] Lim, H. M. (2026). Canonical quantization of a product within
   the Cahill-Glauber correspondence. arXiv:2509.17106v3 [quant-ph]. 
   https://doi.org/10.48550/arXiv.2509.17106

.. [Manzano2020] Manzano, D. (2020). A short introduction to the Lindblad 
   master equation. *AIP Advances*, 10, 025106. 
   https://doi.org/10.1063/1.5115323