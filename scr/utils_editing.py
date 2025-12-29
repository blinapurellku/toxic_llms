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



def get_editing_heads(safe_model_name, output_dir, tox_dir='mean_sv', n=20):
    save_path = os.path.join(output_dir, safe_model_name)
    atten_tensors = load_safetensors(
            os.path.join(save_path, f"attention_states_pure.safetensors")
        )
    
    labels_before = np.load(f"{output_dir}/{safe_model_name}/labels.npy")
    all_head_data = []
   

    for layer_name in list(atten_tensors.keys()):

        overall_mean = atten_tensors[layer_name].float().mean(dim=0)
        toxic_behaviour = atten_tensors[layer_name][labels_before==1].float().mean(dim=0)  # (num_heads, head_dim)
        non_toxic_behaviour = atten_tensors[layer_name][labels_before==0].float().mean(dim=0)  # (num_heads, head_dim)
        head_diff = toxic_behaviour - non_toxic_behaviour  # (num_heads, head_dim)
        
        signed_scores = 1 - F.cosine_similarity(toxic_behaviour, non_toxic_behaviour, dim=-1) #[num_heads]

        r = toxic_behaviour.mean(dim=0) - non_toxic_behaviour.mean(dim=0)
        sims = F.cosine_similarity(atten_tensors[layer_name].float(), r.unsqueeze(0), dim=-1)
        signed_scores = sims.mean(0)       
        
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
    # Process Amplify Heads
    for idx in top_k_amp_indices.tolist():
        layer_name = all_layers[idx]
        head_id = all_head_ids[idx]
        if layer_name not in amplify_heads:
            amplify_heads[layer_name] = []
       
        amplify_heads[layer_name].append(head_id)
          

    print("Top amplify heads:", amplify_heads)

    return amplify_heads