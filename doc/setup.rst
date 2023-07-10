.. _setup:

Setup
=====

The simulation code is written in C++ while the analysis is performed in python3.
All codes are available on GitLab at `this <https://gitlab.com/andrea.ortone/basal_ganglia_model>`_ page.

-   To download the repository run::

         $ git clone https://gitlab.com/andrea.ortone/basal_ganglia_model.git

    Necessary python packages are listed in ``requirements.txt``.

      .. hint:: We recommend working inside a `virtual environment`_; you can create one by executing (from the repository root-directory)::

           $ python3 -m venv <name>
           $ source <name>/bin/activate
           $ pip install -r requirements.txt
           $
           $ ...
           $
           $ deactivate

      .. _virtual environment: https://docs.python.org/3/tutorial/venv.html

-   Necessary C++ include-only libraries are added to the project as git_submodules. In order to use them it is sufficient to run, from the repository root-directory, the command::

        $ git submodule update --init --recursive

-   C++ libraries which have to be built must be separately installed (use apt on linux and brew on MacOS). Required libraries are:

      - the `boost <https://github.com/boostorg/boost>`_ library (basic modules);
      - the `yaml-cpp <https://github.com/jbeder/yaml-cpp>`_ library.

-   C++ code is compiled using cmake. From the ``src`` directory::

        $ mkdir build
        $ cd build
        $ cmake ..
        $ make
