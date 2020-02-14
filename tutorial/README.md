# PIMKL tutorial

## Data retrieval

You can download the data from the tutorial from [here](https://ibm.box.com/v/pimkl-tutorial-data).

In the following we assume you placed files in a folder called data with the following structure:

```console
data
├── data_type_a.csv
├── data_type_b.csv
├── gene_sets.gmt
├── interactions.csv
└── labels.csv

0 directories, 5 files
```

## Installation

For the installation of `pimkl` we suggest to follow the description reported [here](../README.rst).

## Run `pimkl`

```console
pimkl -fd data/data_type_b.csv -nd tutorial --model_name EasyMKL data/interactions.csv network data/gene_sets.gmt genes data/preprocess data/output data/labels.csv
```
