from grape.datasets.kghub import KGIDG
import pandas as pd
from grape.edge_prediction import edge_prediction_evaluation
from grape.edge_prediction import PerceptronEdgePrediction
from grape.embedders import FirstOrderLINEEnsmallen, SecondOrderLINEEnsmallen
from grape.embedders import DeepWalkCBOWEnsmallen, DeepWalkSkipGramEnsmallen
from grape.embedders import WalkletsCBOWEnsmallen, WalkletsSkipGramEnsmallen


g = KGIDG(version='20230601')
main_component = g.remove_components(top_k_components=1)
dense_main_component = main_component.remove_dendritic_trees()


drug_types = ["biolink:ChemicalSubstance", "biolink:ChemicalEntity", "biolink:Drug"]
protein_types = ["biolink:Protein"]
minority_edge_type = 'minority_edge'

dense_main_component.replace_edge_type_name_from_edge_node_type_names_inplace(
    edge_type_name=minority_edge_type,
    source_node_type_names=drug_types,
    destination_node_type_names=protein_types
)

# Set smoke test to True for testing:
SMOKE_TEST = True
NUMBER_OF_HOLDOUTS = 10
VALIDATION_UNBALANCE_RATES = (1.0, )
TRAIN_SIZES = (0.75,)

subgraph = dense_main_component.filter_from_names(
    edge_type_names_to_keep=[minority_edge_type]
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
                        edge_types=[minority_edge_type],
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
results.to_csv("kg_idg_negative_select.tsv",sep="\t")
