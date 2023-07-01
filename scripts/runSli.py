from grape.datasets.kghub import SLDB
from grape.datasets.string import HomoSapiens

from grape.embedders import FirstOrderLINEEnsmallen, SecondOrderLINEEnsmallen
from grape.embedders import DeepWalkGloVeEnsmallen, DeepWalkCBOWEnsmallen, DeepWalkSkipGramEnsmallen
from grape.embedders import WalkletsGloVeEnsmallen, WalkletsCBOWEnsmallen, WalkletsSkipGramEnsmallen
from grape.embedders import HOPEEnsmallen

from grape.edge_prediction import edge_prediction_evaluation
from grape.edge_prediction import PerceptronEdgePrediction

import pandas as pd
import os
from tqdm.auto import tqdm


# Load and process the graphs:
sli_graph = SLDB()
string_graph = HomoSapiens()
graph_root = os.environ.get("GRAPH_CACHE_DIR", "./graphs")

path = f"{graph_root}/string/HomoSapiens/links.v11.5/9606.protein.info.v11.5.txt"
data = pd.read_csv(path, sep="\t")
remapping_string = dict(zip(data[data.columns[0]], data[data.columns[1]]))
string_graph = string_graph     .remove_node_types()     .filter_from_names(min_edge_weight=700)     .remove_edge_weights()     .remap_from_node_names_map(remapping_string)     .set_all_node_types("Gene")     .set_all_edge_types("PPI")     .remove_disconnected_nodes()
path = f"{graph_root}/kghub/SLDB/20220522/sldb/merged-kg_nodes.tsv"
data = pd.read_csv(path, sep="\t")
remapping_sli = dict(zip(data[data.columns[0]], data[data.columns[2]]))

# Restrict to the largest connected component
sli_graph = sli_graph.remap_from_node_names_map(remapping_sli).set_all_node_types("Gene").set_all_edge_types("SLI")
composite_graph = (sli_graph | string_graph).remove_components(top_k_components=1)

composite_graph.enable()

subgraph = composite_graph.filter_from_names(
    edge_type_names_to_keep=["SLI"]
)


# Set smoke test to True for testing:
SMOKE_TEST = False
NUMBER_OF_HOLDOUTS = 10
VALIDATION_UNBALANCE_RATES = (1.0, )
TRAIN_SIZES = (0.75,)


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
            edge_types=["SLI"],
        ),
        evaluation_schema="Connected Monte Carlo",
        graphs=composite_graph,
        models=[
            PerceptronEdgePrediction(
                edge_features=edge_feature,
                number_of_edges_per_mini_batch=32,
                use_scale_free_distribution=use_scale_free_distribution
            )
            for edge_feature in (
                "Degree", "AdamicAdar",
                "JaccardCoefficient", "ResourceAllocationIndex",
                "PreferentialAttachment"
            )
            for use_scale_free_distribution in (True, False)
        ],
        number_of_slurm_nodes=NUMBER_OF_HOLDOUTS,
        enable_cache=True,
        number_of_holdouts=NUMBER_OF_HOLDOUTS,
        use_scale_free_distribution=validation_use_scale_free,
        validation_unbalance_rates=VALIDATION_UNBALANCE_RATES,
        subgraph_of_interest=subgraph,
        use_subgraph_as_support=True
    ))
    for ModelClass in tqdm(
        [
            FirstOrderLINEEnsmallen, SecondOrderLINEEnsmallen,
            DeepWalkGloVeEnsmallen, DeepWalkCBOWEnsmallen, DeepWalkSkipGramEnsmallen,
            WalkletsGloVeEnsmallen, WalkletsCBOWEnsmallen, WalkletsSkipGramEnsmallen,
            HOPEEnsmallen,
        ],
        desc="Embedding",
        leave=False
    ):
        # Do not use scale free for the embedding method, even if it is possible
        results.append(edge_prediction_evaluation(
            smoke_test=SMOKE_TEST,
            holdouts_kwargs=dict(
                train_size=train_size,
                edge_types=["SLI"],
            ),
            evaluation_schema="Connected Monte Carlo",
            node_features=ModelClass(),
            graphs=composite_graph,
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
            number_of_slurm_nodes=NUMBER_OF_HOLDOUTS,
            use_scale_free_distribution=validation_use_scale_free,
            validation_unbalance_rates=VALIDATION_UNBALANCE_RATES,
            subgraph_of_interest=subgraph,
            use_subgraph_as_support=True
        ))

results = pd.concat(results)
results.to_csv("sli_results-hadamard-jul23.tsv",sep="\t")