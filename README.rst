###### 
README
######


This repository contains several notebooks that are designed to demonstrate 
the effect of different strategies of negative example selection on the measured 
performance of graph representation machine learning in which the nodes of a 
knowledge graph are embedded into a low-dimensional vector space, transformed by 
approaches such as the Hadamard transformation into edge representations, following 
which edge classification is performed by methods such as Random Forest or perceptron 
classification.

Graph operations and machine learning are performed 
using  `GRAPE <https://github.com/AnacletoLAB/grape>`_.


Setup
^^^^^

A minimum of Python version 3.8 is required. The easiest way to run the notebooks 
is to set up a virtual environment as follows.

.. code:: bash

    python3 -m venv venv
    source venv/bin/activate
    pip install pandas numpy grape barplots 
    pip install networkx # required for the SliNetwork notebook only
    pip install powerlaw # required for the SliNetwork notebook only
    pip install plotnine # required for some graphics
    pip install jupyter
    python -m ipykernel install --user --name="venv"

Following this, start jupyter or jupyter-notebook and run any of the notebooks in 
this directory.


Biological networks
###################

Biomedical knowledge graphs include
edges representing various types of relations between biological entities such as protein-protein interac-
tions, synthetic lethality, or shared biological functions. Because our knowledge about explicit negative
edges, i.e., pairs of nodes that were demonstrated not to have a specific relation, negative examples are
generally sampled at random from pairs of nodes that are not positively labeled under the assumption
that the majority of non-labelled pairs are negative.

* `SliNetwork <https://github.com/monarch-initiative/negativeExampleSelection/blob/main/SliNetwork.ipynb>`_: Characterization of the Synthetic Lethality Interation (SLI) network.


Classification by Node Degree
#############################

In this experiment, we show that it is possible to obtain respectable classification performance by using solely 
the degree of both nodes of an edge as features.

* `degreeBias <https://github.com/monarch-initiative/negativeExampleSelection/blob/main/degreeBias.ipynb>`_: Classification of the SLI network based on node degree.







Running classification with node-based and edge-based sampling
##############################################################

We ran one analysis on a SLURM cluster that performed analysis 
with different classifiers. We
present the SLURM script we used as well as the python 
script in the ``scripts`` subdirectory.
Running these scripts will generate a folder called ``experiments`` with all the 
results of analysis. The following notebook can be used to ingest these files 
and calculate mean and standard deviations for the various models and parameters.

* `extractingResults <https://github.com/monarch-initiative/negativeExampleSelection/blob/main/extractingResults.ipynb>`_.


We have copied the output of the ``extractingResults`` notebook into the ``results`` folder. 
R script are provided to generate several of the figures in the manuscript.




Model parameters
################

The learning algorithms used for this analysis were implemented in GRAPE and the default parameters
were used. The following script creates a table with these parameters for convenience.

* `modelParameterTable <https://github.com/monarch-initiative/negativeExampleSelection/blob/main/modelParameterTable.ipynb>`_: Extract and summarize parameters.
