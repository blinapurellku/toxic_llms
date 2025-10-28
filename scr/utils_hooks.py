import argparse
import datetime
import gc
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union, Iterable
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from sql_helper import load_prompts_responses, save_prompts_responses

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

torch.set_float32_matmul_precision("high")


import numpy as np
import pandas as pd
from accelerate.utils import find_executable_batch_size
from datasets import load_dataset
from safetensors.torch import save_file as save_safetensors
from utils_templates import LLAMA_CLS_PROMPT, get_template, MISTRAL_CLS_PROMPT
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


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


def steering_vector_hook(
    module: torch.nn.Module,
    steer: torch.Tensor, 
    alpha: float = 1.0, 
    mode: str = "add",
) -> torch.utils.hooks.RemovableHandle:
    """
    Register a forward‐hook on `module` that adds `steer` to its output.
    Returns the hook handle so you can remove it later.
    """
    steer = steer.detach()
    def _hook(_m, _inp, out):
        # Handle HF blocks that return tuples (hidden, present, …)
        x = out[0] if isinstance(out, tuple) else out  # (B, L, H)

        # Broadcast if steer is 1‑D
        add = steer
        if steer.ndim == 1:
            add = steer.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        add = add.to(x.device)

        # if ATTN_MASK is not None:
        #     # ATTN_MASK: shape (B, L) → (B, L, 1)
        #     expanded_mask = ATTN_MASK.unsqueeze(-1).to(x.device)  # (B, L, 1)
        #     add = add * expanded_mask  # (B, L, H) mask-aware addition
        if mode == 'last':
            # Only apply to the last token
            x_last = x[:, -1:]  # (B, 1, H)
            mod_last = x_last + alpha * add  # apply steering to last token

            # Replace last token in x with modified one
            mod = torch.cat([x[:, :-1], mod_last], dim=1)  # (B, L, H)

            # Return in the same format as input
            return (mod,) + out[1:] if isinstance(out, tuple) else mod
        else:
            mod = x + alpha * add
            return (mod,) + out[1:] if isinstance(out, tuple) else mod
        
    return module.register_forward_hook(_hook)


def ablation_hook(
    module: torch.nn.Module,
    head: int = -1,
    num_head: int = 8, # head number of the layer
    ablate: bool = True,
    s_mean : Optional[torch.Tensor] = None,
) -> torch.utils.hooks.RemovableHandle:
    """
    Register a forward pre‐hook on `module` that zeroes out specific indices in its output.
    Returns the hook handle so you can remove it later.
    """
    # s_mean = s_mean.detach()
    s_mean = s_mean.detach() if isinstance(s_mean, torch.Tensor) else None

    def _hook(_m, inp):
        # Handle HF blocks that return tuples (hidden, present, …)
        tgt = inp[0] if isinstance(inp, tuple) else inp  # (B, L, H)
        B, L, _ = tgt.shape
        tgt = tgt.reshape(B, L, num_head, -1)  # (B, L, num_heads, head_dim)
        _, _, _, head_dim = tgt.shape
        if ablate:
            tgt[:, :, head, :].zero_()  # (B, L, H)
        else:
            s_mean = s_mean.to(dtype=tgt.dtype, device=tgt.device)
            tgt[:, :, head, :] = s_mean.unsqueeze(1).expand(B, tgt.size(1), head_dim)  # (B, L, H)
        
        tgt = tgt.reshape(B, L, -1)  # (B, L, H)

        return (tgt,)+ inp[1:] if isinstance(inp, tuple) else tgt
        
    return module.register_forward_pre_hook(_hook)




@torch.no_grad()
def register_head_ablation(
    model: torch.nn.Module,
    spec: Dict[str, Iterable[int]],
    *,
    ablate: bool=True, #str = "zero",                   # "zero" or "fill"
    fill_layer: Optional[Dict[str, torch.Tensor]] = None,  # layers: (head_dim,) or layers: (num_heads, head_dim)
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward pre-hooks on *named* o_proj modules to ablate specific heads.

    Args:
        model: your model.
        spec:  {"model.layers.9.self_attn.o_proj": [1, 3, 5], ...}
        mode:  "zero" → zero the specified heads,
               "fill" → replace with fill_vec.
        fill_vec: replacement vector(s), same logic as before.

    Returns:
        List of hook handles you can remove later.
    """

    handles = []

    def make_hook(num_heads: int, heads: Iterable[int], ablate: bool=True, fill: Optional[torch.Tensor]=None):
        heads = torch.tensor(sorted(set(heads)), dtype=torch.long)

        # if fill is not None:
        #     if fill.dim() == 1:
        #         fill_local = fill.view(1, head_dim).expand(heads.numel(), head_dim).contiguous()
        #     elif fill.dim() == 2:
        #         assert fill.size(0) == heads.numel() and fill.size(1) == head_dim, \
        #             f"fill_vec must be (head_dim,) or ({heads.numel()}, {head_dim})."
        #         fill_local = fill.contiguous()
        #     else:
        #         raise ValueError("fill_vec must be 1D or 2D.")
        # else:
        #     fill_local = None
        fill_norm = fill.detach() if isinstance(fill, torch.Tensor) else None

        def _hook(_m, inp):
            x = inp[0] if isinstance(inp, tuple) else inp  # (B, L, n_heads * head_dim)
            B, L, D = x.shape
           

            x = x.view(B, L, num_heads, -1)
            _,_,_, head_dim = x.shape # (B, L, n_heads, head_dim)

            if ablate: #mode == "zero":
                x.index_fill_(dim=2, index=heads.to(x.device), value=0.0)
            else:
                nonlocal fill_norm
                if fill_norm is None:
                
                    fill_norm = fill.contiguous()
                else:
                    raise ValueError("fill must be 1D or 2D tensor.")
                
                fill_t = fill.to(dtype=x.dtype, device=x.device) # (H, head_dim)

                for idx, h in enumerate(heads.tolist()):
                    # fill_local = fill[h].view(1,1,-1).expand(x.size(0), x.size(1), -1) # (B, L, head_dim)
                    x[:, :, h, :] = fill_t[h].view(1,1,-1).expand(x.size(0), x.size(1), -1) # (B, L, head_dim) fill_local
            # else:
            #     raise ValueError("mode must be 'zero' or 'fill'.")

            x = x.view(B, L, -1).contiguous() # (B, L, H)
            return (x,) + inp[1:] if isinstance(inp, tuple) else x

        return _hook

    for name, heads in spec.items():
        module = dict(model.named_modules()).get(name)
        if module is None:
            raise KeyError(f"Module '{name}' not found in model.named_modules()")

        # Infer number of heads and head_dim from its parent attention module
        # (works for LLaMA-style naming: self_attn.o_proj)
        parent_name = name.rsplit('.', 1)[0]
        parent = dict(model.named_modules()).get(parent_name)
        num_heads = model.config.num_attention_heads #getattr(parent, "num_heads")

        fill = fill_layer.get(name) if fill_layer is not None else None
        
        hook = make_hook(num_heads, heads, ablate, fill)
        handles.append(module.register_forward_pre_hook(hook))
        print(f"✅ Hook registered on {name} for heads {heads}")

    return handles

@contextmanager
def capture_all_layers(model,
                       move_to_cpu: bool = True,
                       pad_and_concat: bool = False,
                       atten: bool = False):
    """
    Record post-block residual streams for *all* decoder layers.

    Yields
    ------
    store : dict[str, list[Tensor] | Tensor]
        While inside the `with`-block a list[Tensor] accumulates per layer.
        On exit, lists are optionally left as-is (*pad_and_concat=False*)
        or left-padded to the layer’s max sequence length and concatenated
        into a single tensor (*pad_and_concat=True*).
    """
    store, handles = defaultdict(list), []

    def _factory(name):
        def _hook(_m, inp, out):
            h = out[0] if isinstance(out, tuple) else out      # (B,L,H)
            h = h.detach().cpu()
            store[name].append(h.bfloat16())
            return out
        return _hook
    
    def _factory_atten(name):
        def _hook_a(_m, inp, out):
            # out: (B, L, H)
            h = inp[0] if isinstance(inp, tuple) else inp      # (B,L,H)
            h = h.detach().cpu()
            store[name].append(h.bfloat16())
            return out
        return _hook_a
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
        layers = [f"model.layers.{i}" for i in range(n)]
        if atten:
            layers = [f"model.layers.{i}.self_attn.o_proj" for i in range(n)]

    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
        layers =  [f"transformer.h.{i}" for i in range(n)]
        if atten:
            layers = [f"transformer.h.{i}.attn.c_proj" for i in range(n)]

    else:
        raise ValueError("Could not determine transformer block count.")

    print(f"Detected {len(layers)} layers: {layers}")
    # Register hooks
    for name, module in model.named_modules():
        if name in layers:
            print(f"Registering hook for {name}")
            if atten:
                handles.append(module.register_forward_hook(_factory_atten(name)))
            else:
                handles.append(module.register_forward_hook(_factory(name)))

    try:
        yield store
    finally:
        for h in handles:
            h.remove()