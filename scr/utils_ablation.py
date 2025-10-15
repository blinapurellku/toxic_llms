import os

import torch
import torch.nn.functional as F

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

torch.set_float32_matmul_precision("high")


import numpy as np
from safetensors.torch import load_file as load_safetensors


# Optional: avoid error spam from Torch Dynamo
torch._dynamo.config.suppress_errors = False

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# torch.use_deterministic_algorithms(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def get_ablation_heads_2(safe_model_name, output_dir, tox_dir='cosine', n=20):
#     save_path = os.path.join(output_dir, safe_model_name)
#     atten_tensors = load_safetensors(
#             os.path.join(save_path, f"attention_states_pure.safetensors")
#         )
    
#     labels_before = np.load(f"{output_dir}/{safe_model_name}/labels.npy")
#     layer_heads = {}
#     all_head_data = []
#     for layer_name in list(atten_tensors.keys()):
#         if layer_name not in layer_heads:
#             layer_heads[layer_name] = {}
        
#         toxic_behaviour = atten_tensors[layer_name][labels_before==1].float().mean(dim=0)  # (num_heads, head_dim)
#         non_toxic_behaviour = atten_tensors[layer_name][labels_before==0].float().mean(dim=0)  # (num_heads, head_dim)
#         head_diff = toxic_behaviour - non_toxic_behaviour  # (num_heads, head_dim)
#         sign_diff = toxic_behaviour.norm(dim=-1) - non_toxic_behaviour.norm(dim=-1)  # (num_heads,)


def get_ablation_heads(safe_model_name, output_dir, tox_dir='pca', n=20):
    save_path = os.path.join(output_dir, safe_model_name)
    atten_tensors = load_safetensors(
            os.path.join(save_path, f"attention_states_pure.safetensors")
        )
    
    labels_before = np.load(f"{output_dir}/{safe_model_name}/labels.npy")
    layer_heads = {}
    all_head_data = []
    # if tox_dir not in ['pca', 'mean_head', 'diff']:
    #     tox_dir = 'mean'


    for layer_name in list(atten_tensors.keys()):
        if layer_name not in layer_heads:
            layer_heads[layer_name] = {}

        overall_mean = atten_tensors[layer_name].float().mean(dim=0)
        toxic_behaviour = atten_tensors[layer_name][labels_before==1].float().mean(dim=0)  # (num_heads, head_dim)
        non_toxic_behaviour = atten_tensors[layer_name][labels_before==0].float().mean(dim=0)  # (num_heads, head_dim)
        head_diff = toxic_behaviour - non_toxic_behaviour  # (num_heads, head_dim)
        if tox_dir == "pca":
            # First principal direction of head_diff (no centering to preserve sign convention)
            _, _, V = torch.pca_lowrank(head_diff, q=1, center=True)
            # _,_, V = torch.pca_lowrank(toxic_behaviour, q=1, center=False)
            tox_axis = V[:, 0]     
            tox_axis = F.normalize(tox_axis, dim=0)          # unit vector
            # head_diff = F.normalize(head_diff, dim=-1)  # unit vectors
            signed_scores = head_diff @ tox_axis  # cosine similarity with tox_axis
       
        elif tox_dir == "mean_head": 
            tox_axis = head_diff.mean(dim=0)  # (head_dim,)
            tox_axis = F.normalize(tox_axis, dim=0)          # unit vector
            # head_diff = F.normalize(head_diff, dim=-1)  # unit vectors
            signed_scores = head_diff @ tox_axis  # cosine similarity with tox_axis

        elif tox_dir == "max_head": 
            tox_axis = head_diff.max(dim=0).values  # (head_dim,)
            tox_axis = F.normalize(tox_axis, dim=0)          # unit vector
            # head_diff = F.normalize(head_diff, dim=-1)  # unit vectors
            signed_scores = head_diff @ tox_axis  # cosine similarity with tox_axis
        
        elif tox_dir== 'cosine_tox':
            signed_scores = F.cosine_similarity(toxic_behaviour, overall_mean, dim=-1) #[num_heads]
        
        elif tox_dir == 'cosine':
            sign = toxic_behaviour.norm(dim=-1) - non_toxic_behaviour.norm(dim=-1)
            sign = torch.sign(sign)
            signed_scores = 1 - F.cosine_similarity(toxic_behaviour, non_toxic_behaviour, dim=-1) #[num_heads]
            signed_scores = signed_scores * sign
            
        elif tox_dir == "diff":
            signed_scores = toxic_behaviour.norm(dim=-1) - non_toxic_behaviour.norm(dim=-1)
        
        else: # this is with mean head
            signed_scores = head_diff.mean(dim=-1)

        amp_idx = torch.nonzero(signed_scores > 0, as_tuple=False).squeeze(1)
        mit_idx = torch.nonzero(signed_scores < 0, as_tuple=False).squeeze(1)

        amplify = amp_idx[torch.argsort(signed_scores[amp_idx], descending=True)[:5]].tolist()
        mitigate = mit_idx[torch.argsort(signed_scores[mit_idx])[:5]].tolist()  # most negative first

        layer_heads[layer_name]['amplify'] = amplify
        layer_heads[layer_name]['mitigate'] = mitigate
        layer_heads[layer_name]['tox_axis'] = toxic_behaviour
        layer_heads[layer_name]['nontox_axis'] = non_toxic_behaviour
        layer_heads[layer_name]['overall'] = overall_mean
        layer_heads[layer_name]['scores'] = signed_scores
        
        head_diff_norms = torch.linalg.norm(head_diff, dim=-1) # (num_heads,)

        for head_id, score in enumerate(signed_scores):
            all_head_data.append({
                'layer': layer_name,
                'head_id': head_id,
                'score': score.item(),
                'diff': head_diff_norms.max().item(),
                })

    all_scores = torch.tensor([d['score'] for d in all_head_data])
    all_layers = [d['layer'] for d in all_head_data]
    all_head_ids = [d['head_id'] for d in all_head_data]

    all_diffs = torch.tensor([d['diff'] for d in all_head_data])

    all_diff_s = torch.sort(all_diffs, descending=True)

    top_k_amp_values, top_k_amp_indices = torch.topk(all_scores, k=min(n, len(all_scores)), largest=True)

    top_k_mit_values, top_k_mit_indices = torch.topk(all_scores, k=min(n, len(all_scores)), largest=False)

    
    amplify_heads = {}
    mitigate_heads = {}
    all_heads = {}
    # Process Amplify Heads
    for idx in top_k_amp_indices.tolist():
        layer_name = all_layers[idx]
        head_id = all_head_ids[idx]
        if layer_name not in amplify_heads:
            amplify_heads[layer_name] = []
        if layer_name not in all_heads:
            all_heads[layer_name] = []
        all_heads[layer_name].append(head_id)
        amplify_heads[layer_name].append(head_id)

    # Process Mitigate Heads
    for idx in top_k_mit_indices.tolist():
        layer_name = all_layers[idx]
        head_id = all_head_ids[idx]
        if layer_name not in mitigate_heads:
            mitigate_heads[layer_name] = []
        if layer_name not in all_heads:
            all_heads[layer_name] = []
        all_heads[layer_name].append(head_id)
        mitigate_heads[layer_name].append(head_id)

    print("Top amplify heads:", amplify_heads)
    print("Top mitigate heads:", mitigate_heads)

    return all_heads, amplify_heads, mitigate_heads, layer_heads