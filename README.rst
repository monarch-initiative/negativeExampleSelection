###### 
README
######


This repository contains several notebooks that are designed to demonstrate the effect of different strategies of negative example selection on the measured performance of graph representation machine learning in which the nodes of a knowledge graph are embedded into a low-dimensional vector space, transformed by approaches such as the Hadamard transformation into edge representations, following which edge classification is performed by methods such as Random Forest or perceptron classification.

Graph operations and machine learning are performed 
using  `GRAPE <https://github.com/AnacletoLAB/grape>`_.


Setup
^^^^^

A minimum of Python version 3.7 is required. The easiest way to run the notebooks is the set up a virtual environment as follows.

.. code:: bash

    python3 -m venv venv
    source venv/bin/activate
    pip install pandas numpy grape barplots
    pip install jupyter
    python -m ipykernel install --user --name=venv

Following this, start jupyter or jupyter-notebook and run any of the notebooks in this directory.


Biological networks
###################

iomedical knowledge graphs include
edges representing various types of relations between biological entities such as protein-protein interac-
tions, synthetic lethality, or shared biological functions. Because our knowledge about explicit negative
edges, i.e., pairs of nodes that were demonstrated not to have a specific relation, negative examples are
generally sampled at random from pairs of nodes that are not positively labeled under the assumption
that the majority of non-labelled pairs are negative.




negativeExampleSelection
Demonstration of bias in graph machine learning owing to negative example selection