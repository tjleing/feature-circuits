import sys
import os
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from nnsight import LanguageModel
from activation_utils import SparseAct
import plotly.graph_objects as go

import torch as t
from typing import Dict, List, Tuple
from collections import defaultdict
from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict

import pandas as pd
import os
from pathlib import Path

DEBUGGING = False
if DEBUGGING:
    tracer_kwargs = {'validate' : True, 'scan' : True}
else:
    tracer_kwargs = {'validate' : False, 'scan' : False}

def read_all_csvs(directory: str, **kwargs) -> dict:
    """
    Read all CSV files from a directory into a dictionary of DataFrames
    
    Args:
        directory: Path to directory containing CSV files
        **kwargs: Additional arguments to pass to pd.read_csv()
    
    Returns:
        Dictionary mapping filenames (without .csv) to DataFrames
    """
    csv_dict = {}
    
    # Convert to Path object for easier handling
    path = Path(directory)
    
    total = []
    # Iterate through all txt files in directory
    for filename in path.glob('*.txt'):
        # Read the txt file
        with open(filename) as file:
            lines = filter(lambda line: line != "", [line.rstrip() for line in file])
            classification = [(line[:-1].rstrip(), line[-1]) for line in lines]
            total += classification
    
    def filter_by_classification(which):
        return list(map(lambda x: x[0], filter(lambda pair: pair[1] == which, total)))

    return [filter_by_classification(which) for which in ['0', '1']]

def analyze_layer_features(
    model,
    dictionaries: Dict,
    submodules: List,
    concept_examples: Dict[str, t.Tensor]
):
    """
    Analyzes how different layers encode conceptual opposites
    
    Args:
        model: The base model
        dictionaries: Dictionary of SAEs per layer
        layer_names: Names/identifiers of layers to analyze
        concept_examples: Dictionary mapping concept names to example inputs
    """
    layer_features = defaultdict(dict)

    with t.no_grad():
        for concept, examples in concept_examples.items():
            for example in examples:
                with model.trace(example, **tracer_kwargs):
                    attn_mask = model.input[1]['attention_mask']
                    for submodule in submodules:
                        if submodule not in dictionaries:
                            print("skipping")
                            continue
                        sae = dictionaries[submodule]

                        if hasattr(submodule, 'attention'):
                            activation = submodule.attention.output
                        if hasattr(submodule, 'mlp'):
                            activation = submodule.mlp.output
                        else:
                            activation = submodule.output

                        if isinstance(activation, tuple):
                            if len(activation) > 0:
                                activation = activation[0] if activation else None
                            else:
                                print(f"Empty tuple for {submodule}")
                                continue

                        # Apply attention mask and average
                        activation = activation * attn_mask[:, :, None]
                        activation = activation.sum(1) / attn_mask.sum(1)[:, None]

                        features = sae.encode(activation)
                        # Average activation pattern for this concept
                        mean_activation = features.mean(dim=0)
                        # Store the most active features for this concept
                        top_features = t.topk(mean_activation, k=5)
                        layer_features[submodule][concept] = {
                            'mean_activation': mean_activation,
                            'top_features': top_features,
                        }
    
    return layer_features

def find_opposite_concepts(
    layer_features: Dict,
    concept1: str,
    concept2: str
) -> Dict:
    """
    Analyzes how different layers encode the opposition between two concepts
    """
    results = {}
    
    for layer, features in layer_features.items():
        c1_activations = features[concept1]['mean_activation']
        c2_activations = features[concept2]['mean_activation']
        
        # Get direction vector between concepts
        concept_direction = c1_activations - c2_activations
        
        # Find features that align most strongly with this direction
        alignment = t.cosine_similarity(
            concept_direction.unsqueeze(0),
            t.eye(len(concept_direction)),
            dim=1
        )
        
        top_aligned = t.topk(alignment, k=5)
        bottom_aligned = t.topk(alignment, k=5, largest=False)
        
        results[layer] = {
            'direction_vector': concept_direction,
            'top_aligned_features': top_aligned,
            'opposite_aligned_features': bottom_aligned,
        }
    
    return results

def analyze_concept_progression(
    layer_features: Dict,
    concept1: str,
    concept2: str
) -> Dict:
    """
    Analyzes how the representation of concept opposites evolves through layers
    """
    progression = {}
    
    for layer in sorted(layer_features.keys()):
        # Get feature separation at this layer
        separation = t.norm(
            layer_features[layer][concept1]['mean_activation'] -
            layer_features[layer][concept2]['mean_activation']
        )
        
        # Get feature sparsity for each concept
        sparsity1 = (layer_features[layer][concept1]['mean_activation'] != 0).float().mean()
        sparsity2 = (layer_features[layer][concept2]['mean_activation'] != 0).float().mean()
        
        progression[layer] = {
            'separation': separation.item(),
            'sparsity': (sparsity1.item(), sparsity2.item()),
        }
    
    return progression

# Assuming you have your model and dictionaries from the attribution code
device = 'cpu' # macos
model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=device, dispatch=True)

# load submodules
submodules = []
start_layer = 0
if start_layer < 0: submodules.append(model.gpt_neox.embed_in)
for i in range(start_layer, len(model.gpt_neox.layers)):
    submodules.extend([
        model.gpt_neox.layers[i].attention,
        model.gpt_neox.layers[i].mlp,
        model.gpt_neox.layers[i]
    ])

submod_names = {
    model.gpt_neox.embed_in : 'embed'
}
for i in range(len(model.gpt_neox.layers)):
    submod_names[model.gpt_neox.layers[i].attention] = f'attn_{i}'
    submod_names[model.gpt_neox.layers[i].mlp] = f'mlp_{i}'
    submod_names[model.gpt_neox.layers[i]] = f'resid_{i}'

dict_id = 10
activation_dim = 512
expansion_factor = 64
dict_size = expansion_factor * activation_dim
feat_dicts = {}
feat_dicts[model.gpt_neox.embed_in] = AutoEncoder.from_pretrained(
    f'../dictionaries/pythia-70m-deduped/embed/{dict_id}_{dict_size}/ae.pt', device=device
)
for i in range(len(model.gpt_neox.layers)):
    feat_dicts[model.gpt_neox.layers[i].attention] = AutoEncoder.from_pretrained(
        f'../dictionaries/pythia-70m-deduped/attn_out_layer{i}/{dict_id}_{dict_size}/ae.pt', device=device
    )
    feat_dicts[model.gpt_neox.layers[i].mlp] = AutoEncoder.from_pretrained(
        f'../dictionaries/pythia-70m-deduped/mlp_out_layer{i}/{dict_id}_{dict_size}/ae.pt', device=device
    )
    feat_dicts[model.gpt_neox.layers[i]] = AutoEncoder.from_pretrained(
        f'../dictionaries/pythia-70m-deduped/resid_out_layer{i}/{dict_id}_{dict_size}/ae.pt', device=device
    )

neuron_dicts = {
    submod : IdentityDict(activation_dim).to(device) for submod in submodules
}

[pos, neg] = list(read_all_csvs('geometry_data'))
concept_examples = {
    'positive': pos,
    'negative': neg,
    # 'formal': get_examples_of_formal_text(),
    # 'casual': get_examples_of_casual_text(),
}
print(analyze_layer_features(model, feat_dicts, submodules, concept_examples))