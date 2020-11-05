PIMKL tutorial
==============

Data retrieval
--------------

You can download the data from the tutorial from
`here <https://ibm.box.com/v/pimkl-tutorial-data>`__.

In the following we assume you placed files in a folder called data with
the following structure:

.. code:: console

    data
    ├── data.csv
    ├── gene_sets.gmt
    ├── interactions.csv
    └── labels.csv

    0 directories, 4 files

Please, check carefully the data format in case you want to run
``pimkl`` on your data.

Installation
------------

For the installation of ``pimkl`` we suggest to follow the description
reported `here <../README.rst>`__.

Run ``pimkl``
-------------

The ``pimkl`` script reproduces the output that can be obtained from the
`PIMKL web service <https://ibm.biz/pimkl-aas>`__.

Pipeline execution
~~~~~~~~~~~~~~~~~~

You can run the full ``pimkl`` pipeline (supervised) by executing:

.. code:: console

    pimkl -fd data/data.csv -nd tutorial --model_name EasyMKL data/interactions.csv network data/gene_sets.gmt genes data/preprocess data/output data/labels.csv

You can change the parameters (e.g., regularization, number of folds) by
providing them to the script:

.. code:: console

    pimkl --help
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

For example:

.. code:: console

    pimkl -fd data/data.csv -nd tutorial --model_name EasyMKL data/interactions.csv network data/gene_sets.gmt genes data/preprocess data/output data/labels.csv 0.2 5 50

**Tip:** increasing the number of folds is useful to have better
estimates of the significant gene sets/pathways.

Stepwise execution
~~~~~~~~~~~~~~~~~~

``pimkl`` pipeline can be also run in a stepwise fashion.

Preprocessing
^^^^^^^^^^^^^

``pimkl-preprocess`` prepares the pathway-specific inducers and the data
for the subsequent analysis:

.. code:: console

    pimkl-preprocess --help
    Usage: pimkl-preprocess [OPTIONS] NETWORK_CSV_FILE NETWORK_NAME
                            GENE_SETS_GMT_FILE GENE_SETS_NAME PREPROCESS_DIR

      Compute incuding Laplacian matrices and preprocess data matrices for
      matching features.

      Multiple data_csv_files may be passed. Each data_csv_file should readable
      as pandas.DataFrames `pd.read_csv(filename, sep=',', index_col=0)` where
      index are features (rows) and columns a are samples.

      The `network_csv_file` is an edge list readable with
      `pd.read_csv(filename)` where the 3rd columns is a numeric value.

      The `gene_sets_gmt_file` should follow the gmt specification. See http://s
      oftware.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_format
      s

      For each file, a name has to be passed. Names cannot contain "_" or "-".

      Results are written to `preprocess_dir`.

    Options:
      -fd, --data_csv_file PATH  [required]
      -nd, --data_name TEXT      [required]
      --help                     Show this message and exit.

Execute it on the tutorial data by running:

.. code:: console

    pimkl-preprocess -fd data/data.csv -nd tutorial data/interactions.csv network data/gene_sets.gmt genes data/preprocess

Analysis
^^^^^^^^

``pimkl-analyse`` is responsible of analysing the preprocessed data.

.. code:: console

    pimkl-analyse --help
    Usage: pimkl-analyse [OPTIONS] COMMAND [ARGS]...

    Options:
      --help  Show this message and exit.

    Commands:
      run-kpca                  KernelPCA with given pathway weights
      run-performance-analysis  train and test many folds

Here we focus on the component
``pimkl-analyse run-performance-analysis``, to obtain prediction
performance and an estimate of the most significant gene sets/pathways:

.. code:: console

    pimkl-analyse run-performance-analysis --help
    Usage: pimkl-analyse run-performance-analysis [OPTIONS] NETWORK_NAME
                                                  GENE_SETS_NAME PREPROCESS_DIR
                                                  OUTPUT_DIR CLASS_LABEL_FILE
                                                  [LAM] [K] [NUMBER_OF_FOLDS]
                                                  [MAX_PER_CLASS] [SEED]
                                                  [MAX_PROCESSES]

      Run classifications using pathway induced multiple kernel learning on
      preprocessed data and inducers on a number of train/test splits and
      analyse the resulting classification performance and learned pathway
      weights.

      The `class_label_file` should be readable with `pd.read_csv(
      class_label_file, index_col=0, header=None, squeeze=True)`

    Options:
      -nd, --data_name TEXT           [required]
      --model_name [EasyMKL|UMKLKNN|AverageMKL]
      --help                          Show this message and exit.

Run the analysis on the tutorial data by executing:

.. code:: console

    pimkl-analyse run-performance-analysis -nd tutorial --model_name EasyMKL network genes data/preprocess data/output data/labels.csv

If you want to have more info/examples on how to use ``pimkl`` feel free
to open an issue with the tag
`tutorial <https://github.com/PhosphorylatedRabbits/pimkl/labels/tutorial>`__
on the repo.
