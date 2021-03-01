#!/usr/bin/env python
# coding: utf-8

# # AD multiomic analysis
# 
# PIMKL analysis of the data presented in https://doi.org/10.1016/j.nbd.2018.12.009.

# In[ ]:


import os
import pandas as pd
from pimkl.inducers import read_gmt, get_matching_data_and_network, write_preprocessed
from pimkl.network import get_network_from_csv
from pimkl.cli.analyse import analyse, sns, plt, significant_color, other_color, significant_pathways


# In[ ]:


get_ipython().system('pip install pyxlsb')
get_ipython().system('pip install xlrd')


# In[ ]:


# paths and setup
data_path = os.path.join(os.path.expanduser('~'), 'path/to/data')
proteomic_filepath = os.path.join(data_path, '30-058_Protein_210111.xlsb')
transcriptomic_filepath = os.path.join(data_path, '30-058_Transcriptome_210111.xlsb')
gene_sets_filepath = os.path.join(data_path, 'h.all.v7.2.symbols.xls')
network_filepath = os.path.join(data_path, 'kegg_pc_interactions.csv')
preprocess_dir = os.path.join(data_path, 'preprocess')
os.makedirs(preprocess_dir, exist_ok=True)
network_name = os.path.splitext(os.path.basename(network_filepath))[0].replace('_', '-')
gene_sets_name = os.path.splitext(os.path.basename(gene_sets_filepath))[0].replace('.', '-')
output_dir = os.path.join(data_path, 'output')
os.makedirs(output_dir, exist_ok=True)
labels_filepath = os.path.join(preprocess_dir, 'labels.csv')
data_names = ['proteome', 'transcriptome']


# In[ ]:


def read_expression_data(filepath, gene_name_column, entities_start_index, samples_start_index, samples_end_index):
    data = pd.read_excel(filepath, engine='pyxlsb')
    gene_names = data[gene_name_column][entities_start_index:].str.strip('[]')
    expression = data[entities_start_index:].T[samples_start_index:samples_end_index]
    expression.columns = gene_names
    expression.index.name = 'samples'
    expression.columns.name = 'gene_names'
    return ((expression - expression.mean()) / expression.std()).fillna(0.)


def read_stages(transcriptomic_filepath):
    data = pd.read_excel(transcriptomic_filepath, engine='pyxlsb')
    return data.iloc[0][2:].astype(int)


def plot_top_k_pathways(weights_filepath, top_k):
    analysis_name = os.path.splitext(os.path.basename(weights_filepath))[0].replace('weights_', '')
    weights_df = pd.read_csv(weights_filepath, index_col=0)
    inducers_ordering = weights_df.median().sort_values(ascending=False).index
    colors = [
        significant_color if is_significant else other_color for is_significant
        in significant_pathways(weights_df[inducers_ordering])
    ]
    sns.boxplot(data=weights_df[inducers_ordering[:top_k]], palette=colors[:top_k])
    plt.xlabel('Pathway')
    plt.ylabel('Weight')
    _ = plt.xticks(rotation=90, fontsize=8)
    plt.axhline(y=1. / weights_df.shape[1], lw=1., ls='--', c='black')
    plt.savefig(
        '{}/weights_top-k={}_{}.pdf'.format(output_dir, top_k, analysis_name),
        bbox_inches='tight'
    )
    plt.close()


# In[ ]:


# read data
proteomic_data = read_expression_data(proteomic_filepath, 'Class', 3, 4, 40)
transcriptomic_data = read_expression_data(transcriptomic_filepath, 'Class.1', 4, 2, 62)
stages = read_stages(transcriptomic_filepath)
gene_sets = read_gmt(gene_sets_filepath)
network = get_network_from_csv(network_filepath)


# In[ ]:


# write inducers
for data_name, data in zip(data_names, [proteomic_data, transcriptomic_data]):
    data_filtered, network_filtered = get_matching_data_and_network(
        data, network
    )
    write_preprocessed(
        data=data_filtered,
        data_name=data_name,
        network=network_filtered,
        network_name=network_name,
        gene_sets=gene_sets,
        gene_sets_name=gene_sets_name,
        output_dir=preprocess_dir
    )


# In[ ]:


# write labels file
# focusing on low versus high
(stages > 3).astype(int).to_csv(labels_filepath, header=None)


# In[ ]:


# analysis with supervised PIMKL (EasyMKL)
analysis_name = analyse(
    data_names, network_name, gene_sets_name,
    preprocess_dir, output_dir,
    class_label_file=labels_filepath, model_name='EasyMKL',
    number_of_folds=25,
    max_per_class=10
)


# In[ ]:


# supervised weights detail
weights_filepath = os.path.join(output_dir, 'weights_{}.csv'.format(analysis_name))
plot_top_k_pathways(weights_filepath, top_k=50)


# In[ ]:


# analysis with unsupervised PIMKL (UMKLKNN)
analysis_name = analyse(
    data_names, network_name, gene_sets_name,
    preprocess_dir, output_dir,
    class_label_file=labels_filepath, model_name='UMKLKNN',
    number_of_folds=25,
    max_per_class=10
)


# In[ ]:


# unsupervised weights detail
weights_filepath = os.path.join(output_dir, 'weights_{}.csv'.format(analysis_name))
plot_top_k_pathways(weights_filepath, top_k=50)


# In[ ]:




