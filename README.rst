=====
pimkl
=====


.. .. image:: https://travis-ci.org/PhosphorylatedRabbits/pimkl.svg
    :target: https://travis-ci.org/PhosphorylatedRabbits/pimkl

.. .. image:: https://readthedocs.org/projects/pimkl/badge/?version=latest
        :target: https://pimkl.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. .. image:: https://pyup.io/repos/github/PhosphorylatedRabbits/pimkl/shield.svg
     :target: https://pyup.io/repos/github/PhosphorylatedRabbits/pimkl/
     :alt: Updates



pathway induced multiple kernel learning for computational biology


* Free software: MIT license
* Documentation: https://pimkl.readthedocs.io.


Features
--------

The pimkl command:

.. code-block:: console

    Usage: pimkl [OPTIONS] NETWORK_CSV_FILE NETWORK_NAME GENE_SETS_GMT_FILE
             GENE_SETS_NAME PREPROCESS_DIR OUTPUT_DIR CLASS_LABEL_FILE [LAM]
             [K] [NUMBER_OF_FOLDS] [MAX_PER_CLASS] [SEED] [MAX_PROCESSES]
             [FOLD]

    Console script for a complete pimkl pipeline, including preprocessing and
    analysis. For more details consult the following console scripts, which
    are here executed in this order. `pimkl-preprocess --help` `pimkl-analyse
    run-performance-analysis --help`

    Options:
    -fd, --data_csv_file PATH       [required]
    -nd, --data_name TEXT           [required]
    --model_name [EasyMKL|UMKLKNN|AverageMKL]
    --help                          Show this message and exit.

Requirements
-------------

* C++14 capable C++ compiler
* cmake (>3.0.2)
* Python


Installation
-------------

Install the dependencies

.. code-block:: bash

    pip install -r requirements.txt

Install the package

.. code-block:: bash

    pip install .


Tutorial
---------

You can find a brief tutorial in the dedicated folder.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
