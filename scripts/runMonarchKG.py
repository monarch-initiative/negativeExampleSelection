from grape.datasets.kghub import KGIDG
import pandas as pd
from grape.edge_prediction import edge_prediction_evaluation
from grape.edge_prediction import PerceptronEdgePrediction
from grape.embedders import FirstOrderLINEEnsmallen, SecondOrderLINEEnsmallen
from grape.embedders import DeepWalkCBOWEnsmallen, DeepWalkSkipGramEnsmallen
from grape.embedders import WalkletsCBOWEnsmallen, WalkletsSkipGramEnsmallen


import os

import requests
fname = 'guppy-0.1.10.tar.gz'
url = 'https://pypi.python.org/packages/source/g/guppy/' + fname



edges_fname = "monarch-kg_edges.tsv"
nodes_fname = "monarch-kg_nodes.tsv"
if not os.path.exists(edges_fname) or not os.path.exists(nodes_fname):
    print("Need to download and unpack Monarch KG before running this script")
    print("!wget https://data.monarchinitiative.org/monarch-kg-dev/2023-09-28/monarch-kg.tar.gz")
    print("tar -xzvf monarch-kg.tar.gz")
    exit(1)


from grape import Graph
mg = Graph.from_csv(
          directed=False,
          node_path=nodes_fname,
          edge_path=edges_fname,
          verbose=True,
          nodes_column='id',
          node_list_node_types_column='category',
          default_node_type='biolink:NamedThing',
          sources_column='subject',
          destinations_column='object',
          edge_list_edge_types_column='predicate',
          name='kg-monarch-2022-12-11')

# Extract main component
main_component = mg.remove_components(top_k_components=1)
dense_main_component = main_component.remove_dendritic_trees()

disease_types = ['biolink:Disease']
phenotype_types = ["biolink:PhenotypicFeature"]
d2p_edge_type = 'biolink:has_phenotype' # disease to phenotype
dense_main_component = dense_main_component.remove_parallel_edges()

dense_main_component.replace_edge_type_name_from_edge_node_type_names_inplace(
    edge_type_name=d2p_edge_type,
    source_node_type_names=phenotype_types,
    destination_node_type_names=disease_types
)

# Set smoke test to True for testing:
SMOKE_TEST = True
NUMBER_OF_HOLDOUTS = 10
VALIDATION_UNBALANCE_RATES = (1.0, )
TRAIN_SIZES = (0.75,)

subgraph = dense_main_component.filter_from_names(
    edge_type_names_to_keep=[d2p_edge_type]
)

# Disable the smoke test when you need to run the real thing:

SMOKE_TEST = True
NUMBER_OF_HOLDOUTS = 10
VALIDATION_UNBALANCE_RATES = (1.0, )
TRAIN_SIZES = (0.75,)

results = []

for train_size in TRAIN_SIZES:
    for validation_use_scale_free in (True, False):
        for ModelClass in  [
                FirstOrderLINEEnsmallen, SecondOrderLINEEnsmallen,
                DeepWalkCBOWEnsmallen, DeepWalkSkipGramEnsmallen,
                WalkletsCBOWEnsmallen, WalkletsSkipGramEnsmallen,
            ]:
            # If the embedding method involves edge sampling, we train a run
            # using the scale free and one using the uniform.
            if "use_scale_free_distribution" in ModelClass().parameters():
                parameter_sets = [
                    dict(
                        use_scale_free_distribution = True
                    ),
                    dict(
                        use_scale_free_distribution = False
                    )
                ]
            else:
                parameter_sets = [dict()]
            print(f"train size: {train_size}; validation_use_scale_free: {validation_use_scale_free} ModelClass {ModelClass}")
            for parameter_set in parameter_sets:
                results.append(edge_prediction_evaluation(
                    smoke_test=SMOKE_TEST,
                    holdouts_kwargs=dict(
                        train_size=train_size,
                        edge_types=[d2p_edge_type],
                    ),
                    evaluation_schema="Connected Monte Carlo",
                    node_features=ModelClass(**parameter_set),
                    graphs=dense_main_component,
                    models=[
                        PerceptronEdgePrediction(
                            edge_features=None,
                            edge_embeddings="Hadamard",
                            number_of_edges_per_mini_batch=32,
                            use_scale_free_distribution=use_scale_free_distribution
                        )
                        for use_scale_free_distribution in (True, False)
                    ],
                    enable_cache=True,
                    number_of_holdouts=NUMBER_OF_HOLDOUTS,
                    use_scale_free_distribution=validation_use_scale_free,
                    validation_unbalance_rates=VALIDATION_UNBALANCE_RATES,
                    subgraph_of_interest=subgraph,
                    use_subgraph_as_support=False
                ))

results = pd.concat(results)
results.to_csv("d2p_negative_select_oct30.tsv",sep="\t")