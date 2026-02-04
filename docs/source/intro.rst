Introduction
############

Let's start introducing the methematical definition of ε-differential privacy:

.. _definition-epsilon-dp:

**Definition (ε-differential privacy).**

A randomized algorithm :math:`\mathcal{M}`, with domain :math:`\mathcal{D}` and range
:math:`\mathcal{R}`, satisfies **ε-differential privacy** if for any two adjacent inputs
:math:`Y, Y' \in \mathcal{D}` and for any subset of outputs
:math:`S \subseteq \mathcal{R}` it holds that

.. math::

   \mathbb{P}[\mathcal{M}(Y)\in S]
   \leq e^{\epsilon}\,\mathbb{P}[\mathcal{M}(Y')\in S],

with :math:`\epsilon \geq 0`.

**Definition (ε-differential privacy).**

A randomized algorithm :math:`\mathcal{M}`, with domain :math:`\mathcal{D}` and range
:math:`\mathcal{R}`, satisfies **ε-differential privacy** if for any two adjacent inputs
:math:`Y, Y' \in \mathcal{D}` and for any subset of outputs
:math:`S \subseteq \mathcal{R}` it holds that

.. math::

   \mathbb{P}[\mathcal{M}(Y)\in S]
   \leq e^{\epsilon}\,\mathbb{P}[\mathcal{M}(Y')\in S],

with :math:`\epsilon \geq 0`.

In this definition the value of :math:`\epsilon` is the privacy budget, which is the parameter used to control the level of privacy. 

Then, we can introduce the definition of (ε, δ)-differential privacy), which incoporated a parameter :math:`\delta` that represents the probability of exceeding the privacy budget:

.. _definition-epsilon-delta-dp:

**Definition ((ε, δ)-differential privacy).**

Let :math:`\mathcal{M}` be a randomized algorithm with domain
:math:`\mathcal{D}` and range :math:`\mathcal{R}`. It satisfies
**(ε, δ)-differential privacy** if for any
two adjacent inputs :math:`Y, Y' \in \mathcal{D}` and for any subset of outputs
:math:`S \subseteq \mathcal{R}` it holds that

.. math::

   \mathbb{P}[\mathcal{M}(Y)\in S]
   \leq e^{\epsilon}\,\mathbb{P}[\mathcal{M}(Y')\in S] + \delta,

with :math:`\epsilon \geq 0` and :math:`\delta \in [0,1]`.


