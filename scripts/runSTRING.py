
import platform
my_version = platform.python_version()
assert my_version == "3.8.17", my_version



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

NUMBER_OF_HOLDOUTS = 10
SMOKE_TEST = False

# Load and process the graphs:

string_graph = HomoSapiens()
graph_root = os.environ.get("GRAPH_CACHE_DIR", "./graphs")

path = f"{graph_root}/string/HomoSapiens/links.v11.5/9606.protein.info.v11.5.txt"
data = pd.read_csv(path, sep="\t")
string_graph = string_graph.remove_node_types().filter_from_names(min_edge_weight=700).remove_edge_weights().remove_disconnected_nodes()


results = []

fresults = []
train_size = 0.75
for validation_use_scale_free in tqdm(
    (True, False),
    desc="Validation use scale free",
    leave=False
):
    for ModelClass in tqdm(
            [
                FirstOrderLINEEnsmallen, SecondOrderLINEEnsmallen,
                DeepWalkCBOWEnsmallen, DeepWalkSkipGramEnsmallen,
                WalkletsCBOWEnsmallen, WalkletsSkipGramEnsmallen,
            ],
            desc="Embedding",
            leave=False
    ):
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
            
        for parameter_set in parameter_sets:
            results.append(edge_prediction_evaluation(
                smoke_test=SMOKE_TEST,
                holdouts_kwargs=dict(
                    train_size=train_size,
                    #edge_types=["SLI"],
                ),
                evaluation_schema="Connected Monte Carlo",
                node_features=ModelClass(**parameter_set),
                graphs=string_graph,
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
                validation_unbalance_rates=[1.0],
            ))
                
results = pd.concat(results)

if not SMOKE_TEST:
    outfilename = f"string_only_results_jul23.tsv" 
    results.to_csv(outfilename, sep="\t")
else:
    outfilename = f"string_only_smoke_test.tsv" 
    results.to_csv(outfilename, sep="\t")
