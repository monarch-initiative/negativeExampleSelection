from grape.datasets.kghub import KGIDG
from tqdm import tqdm
import pandas as pd
from grape.edge_prediction import edge_prediction_evaluation
from grape.edge_prediction import PerceptronEdgePrediction



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
SMOKE_TEST = False
NUMBER_OF_HOLDOUTS = 10
VALIDATION_UNBALANCE_RATES = (1.0, )
TRAIN_SIZES = (0.75,)

subgraph = dense_main_component.filter_from_names(
    edge_type_names_to_keep=["minority_edge"]
)

results = []

fresults = []
train_size = 0.75
for validation_use_scale_free in tqdm(
    (True, False),
    desc="Validation use scale free",
    leave=False
    ):
    results.append(edge_prediction_evaluation(
        smoke_test=SMOKE_TEST,
        holdouts_kwargs=dict(
            train_size=train_size,
            edge_types=["minority_edge"],
        ),
        evaluation_schema="Connected Monte Carlo",
        graphs=dense_main_component,
        models=[
            PerceptronEdgePrediction(
                edge_features=edge_feature,
                number_of_epochs=1000,
                number_of_edges_per_mini_batch=16,
                learning_rate=0.001,
                use_scale_free_distribution=False
            ) for edge_feature in("Degree","AdamicAdar","JaccardCoefficient","ResourceAllocationIndex","PreferentialAttachment")
        ],
        #number_of_slurm_nodes=NUMBER_OF_HOLDOUTS,
        enable_cache=True,
        number_of_holdouts=NUMBER_OF_HOLDOUTS,
        use_scale_free_distribution=validation_use_scale_free,
        validation_unbalance_rates=VALIDATION_UNBALANCE_RATES,
        subgraph_of_interest=subgraph,
        use_subgraph_as_support=True
    ))
results = pd.concat(results)
results.to_csv("kg_idg_negative_select.tsv",sep="\t")
