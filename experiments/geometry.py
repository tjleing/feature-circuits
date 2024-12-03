import sys
import os
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from nnsight import LanguageModel
from nnsight.envoy import Envoy
from activation_utils import SparseAct
import plotly.graph_objects as go

import torch as t
import torch.nn as nn
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
    concept_examples: Dict[str, List[str]]
):
    """
    Analyzes how different layers encode conceptual opposites
    """
    layer_features = defaultdict(dict)

    with t.no_grad():
        for concept, examples in concept_examples.items():
            print(f"\nProcessing concept: {concept}")
            accumulated_features = defaultdict(list)
            
            for example in examples:
                try:
                    with model.trace(example, **tracer_kwargs):
                        attn_mask = model.input[1]['attention_mask']
                        
                        for submodule in submodules:
                            if submodule not in dictionaries:
                                continue
                                
                            # Debug print
                            print(f"\nProcessing submodule: {type(submodule).__name__}")
                            
                            # Get activation based on submodule type
                            try:
                                # First try direct output access
                                activation = submodule.output
                                print("Got output directly")
                            except Exception as e:
                                print(f"Direct output access failed: {e}")
                                # If that fails, try to get it from the layer
                                layer_num = None
                                for i in range(len(model.gpt_neox.layers)):
                                    if any(submodule is x for x in [
                                        model.gpt_neox.layers[i],
                                        model.gpt_neox.layers[i].attention,
                                        model.gpt_neox.layers[i].mlp
                                    ]):
                                        layer_num = i
                                        break
                                
                                if layer_num is not None:
                                    print(f"Found in layer {layer_num}")
                                    if isinstance(submodule, type(model.gpt_neox.layers[layer_num].attention)):
                                        activation = model.gpt_neox.layers[layer_num].attention.output
                                    elif isinstance(submodule, type(model.gpt_neox.layers[layer_num].mlp)):
                                        activation = model.gpt_neox.layers[layer_num].mlp.output
                                    else:
                                        activation = model.gpt_neox.layers[layer_num].output
                                else:
                                    # Handle embedding layer
                                    activation = model.gpt_neox.embed_in.output
                            
                            # Debug print for activation
                            print(f"Activation type: {type(activation)}")
                            if isinstance(activation, tuple):
                                print(f"Tuple length: {len(activation)}")
                            
                            # Process activation
                            if isinstance(activation, tuple):
                                activation = activation[0]
                            
                            # Apply attention mask
                            activation = activation * attn_mask[:, :, None]
                            activation = activation.sum(1) / attn_mask.sum(1)[:, None]
                            
                            # Get features
                            sae = dictionaries[submodule]
                            features = sae.encode(activation)
                            accumulated_features[submodule].append(features)
                            
                except Exception as e:
                    print(f"Error processing example '{example[:50]}...': {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Process accumulated features
            for submodule in accumulated_features:
                if not accumulated_features[submodule]:
                    continue
                
                all_features = t.stack(accumulated_features[submodule])
                mean_activation = all_features.mean(dim=0)
                top_features = t.topk(mean_activation, k=5)
                
                layer_features[submodule][concept] = {
                    'mean_activation': mean_activation,
                    'top_features': top_features,
                }
    
    return layer_features

# def analyze_layer_features(
#     model,
#     dictionaries: Dict,
#     submodules: List,
#     concept_examples: Dict[str, t.Tensor]
# ):
#     """
#     Analyzes how different layers encode conceptual opposites
    
#     Args:
#         model: The base model
#         dictionaries: Dictionary of SAEs per layer
#         layer_names: Names/identifiers of layers to analyze
#         concept_examples: Dictionary mapping concept names to example inputs
#     """
#     layer_features = defaultdict(dict)

#     with t.no_grad():
#         for concept, examples in concept_examples.items():
#             for example in examples:
#                 with model.trace(example, **tracer_kwargs):
#                     # print(model.input)
#                     # print(model.input[1])
#                     # print(model.input[1]['attention_mask'])
#                     attn_mask = model.input[1]['attention_mask']
#                     for submodule in submodules:
#                         if submodule not in dictionaries:
#                             print("skipping")
#                             continue
#                         if isinstance(submodule, Envoy):
#                             submodule = submodule.target

#                         print(f"\nProcessing submodule: {type(submodule).__name__}")
#                         layer_num = None
#                         for i in range(len(model.gpt_neox.layers)):
#                             if submodule in [
#                                 model.gpt_neox.layers[i],
#                                 model.gpt_neox.layers[i].attention,
#                                 model.gpt_neox.layers[i].mlp
#                             ]:
#                                 layer_num = i
#                                 break


#                         if layer_num is not None:
#                             if hasattr(submodule, 'attention'):
#                                 activation = model.gpt_neox.layers[layer_num].attention.output[0]
#                             elif hasattr(submodule, 'mlp'):
#                                 activation = model.gpt_neox.layers[layer_num].mlp.output[0]
#                             else:
#                                 activation = model.gpt_neox.layers[layer_num].output[0]
#                         else:
#                             # Handle embedding layer
#                             activation = submodule.output[0]
                        
#                         print(f"Activation type: {type(activation)}")
#                         if isinstance(activation, tuple):
#                             print(f"Tuple length: {len(activation)}")
#                             activation = activation[0]

#                         activation = activation * attn_mask[:, :, None]
#                         activation = activation.sum(1) / attn_mask.sum(1)[:, None]

#                         sae = dictionaries[submodule]
#                         features = sae.encode(activation)
#                         # Average activation pattern for this concept
#                         mean_activation = features.mean(dim=0)
#                         # Store the most active features for this concept
#                         top_features = t.topk(mean_activation, k=5)
#                         layer_features[submodule][concept] = {
#                             'mean_activation': mean_activation,
#                             'top_features': top_features,
#                         }

    
#     return layer_features

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
for subm in submodules:
    print(subm, type(subm))
# print(analyze_layer_features(model, feat_dicts, submodules, concept_examples))



##### from bib_circuit.ipynb

layer = 4 # the model layer to attach linear classification head to
SEED = 42

from datasets import load_dataset
import random
dataset = load_dataset("LabHC/bias_in_bios")
profession_dict = {'professor' : 21, 'nurse' : 13}
male_prof = 'professor'
female_prof = 'nurse'

batch_size = 1024
DEVICE = device
SEED = 42

def get_data(train=True, ambiguous=True, batch_size=128, seed=SEED):
    if train:
        data = dataset['train']
    else:
        data = dataset['test']
    if ambiguous:
        neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 0]
        pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 1]
        n = min([len(neg), len(pos)])
        neg, pos = neg[:n], pos[:n]
        data = neg + pos
        labels = [0]*n + [1]*n
        idxs = list(range(2*n))
        random.Random(seed).shuffle(idxs)
        data, labels = [data[i] for i in idxs], [labels[i] for i in idxs]
        true_labels = spurious_labels = labels
    else:
        neg_neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 0]
        neg_pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 1]
        pos_neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 0]
        pos_pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 1]
        n = min([len(neg_neg), len(neg_pos), len(pos_neg), len(pos_pos)])
        neg_neg, neg_pos, pos_neg, pos_pos = neg_neg[:n], neg_pos[:n], pos_neg[:n], pos_pos[:n]
        data = neg_neg + neg_pos + pos_neg + pos_pos
        true_labels     = [0]*n + [0]*n + [1]*n + [1]*n
        spurious_labels = [0]*n + [1]*n + [0]*n + [1]*n
        idxs = list(range(4*n))
        random.Random(seed).shuffle(idxs)
        data, true_labels, spurious_labels = [data[i] for i in idxs], [true_labels[i] for i in idxs], [spurious_labels[i] for i in idxs]

    batches = [
        (data[i:i+batch_size], t.tensor(true_labels[i:i+batch_size], device=DEVICE), t.tensor(spurious_labels[i:i+batch_size], device=DEVICE)) for i in range(0, len(data), batch_size)
    ]

    return batches

# def get_data():
#     batch_size = 100
#     data = pos # = neg
#     batches = [
#         (data[i:i+batch_size], t.tensor([1] * batch_size, device=device), t.tensor([0] * batch_size, device=device)) for i in range(0, len(data), batch_size)
#     ]

#     return batches
class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits

def train_probe(get_acts, label_idx=0, batches=get_data(), lr=1e-2, epochs=1, dim=512, seed=SEED):
    t.manual_seed(seed)
    probe = Probe(dim).to('cpu')
    optimizer = t.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    for epoch in range(epochs):
        for batch in batches:
            text = batch[0]
            labels = batch[label_idx+1] 
            acts = get_acts(text)
            logits = probe(acts)
            loss = criterion(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return probe, losses

def test_probe(probe, get_acts, label_idx=0, batches=get_data(), seed=SEED):
    with t.no_grad():
        corrects = []

        for batch in batches:
            text = batch[0]
            labels = batch[label_idx+1]
            acts = get_acts(text)
            logits = probe(acts)
            preds = (logits > 0.0).long()
            corrects.append((preds == labels).float())
        return t.cat(corrects).mean().item()
    
def get_acts(text):
    with t.no_grad(): 
        with model.trace(text, **tracer_kwargs):
            attn_mask = model.input[1]['attention_mask']
            acts = model.gpt_neox.layers[layer].output[0]
            acts = acts * attn_mask[:, :, None]
            acts = acts.sum(1) / attn_mask.sum(1)[:, None]
            acts = acts.save()
        return acts.value

probe, _ = train_probe(get_acts, label_idx=0)
print('Ambiguous test accuracy:', test_probe(probe, get_acts, label_idx=0))
batches = get_data()
print('Ground truth accuracy:', test_probe(probe, get_acts, batches=batches, label_idx=0))
print('Unintended feature accuracy:', test_probe(probe, get_acts, batches=batches, label_idx=1))