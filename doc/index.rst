.. basal_ganglia_model_doc documentation master file, created by
   sphinx-quickstart on Mon Oct 11 10:40:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Basal Ganglia model's documentation
==============================================



The purpose of the project that is here documented is the simulation of a model for the Basal Ganglia Network.
The code has been developed in such a way that general networks of supported point neurons can be simulated.

Supported neuron models are (name-conventions of `NEST-simulator <https://nest-simulator.readthedocs.io/en/v3.3/contents.html>`_ are adopted, when possible):

- `iaf_cond_exp <https://nest-simulator.readthedocs.io/en/v3.3/models/iaf_cond_exp.html?highlight=iaf_cond_exp>`_
- `iaf_cond_alpha <https://nest-simulator.readthedocs.io/en/v3.3/models/iaf_cond_alpha.html?highlight=iaf_cond_alpha>`_
- `aeif_cond_exp <https://nest-simulator.readthedocs.io/en/v3.3/models/aeif_cond_exp.html?highlight=aeif_cond_exp>`_
- aqif_cond_exp
- aqif2_cond_exp

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Table of Contents
-----------------
.. toctree::
    :maxdepth: 2

    setup
    api/index
    example
