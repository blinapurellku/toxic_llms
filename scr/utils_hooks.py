import argparse
import datetime
import gc
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union
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
from utils_evaluating_toxicity import classify_generation
from utils_load_dataset_and_models import load_model_and_tokenizer, load_classifier, load_dataset, classify_models_dict
from generate_responses import classify_generation, generate_responses


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
) -> torch.utils.hooks.RemovableHandle:
    """
    Register a forward‐hook on `module` that adds `steer` to its output.
    Returns the hook handle so you can remove it later.
    """
    steer = steer.detach()
    def _hook(_mod, _inp, out):
        # Handle HF blocks that return tuples (hidden, present, …)
        tgt = out[0] if isinstance(out, tuple) else out  # (B, L, H)

        # Broadcast if steer is 1‑D
        add = steer
        if steer.ndim == 1:
            add = steer.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        add = add.to(tgt.device)

        # if ATTN_MASK is not None:
        #     # ATTN_MASK: shape (B, L) → (B, L, 1)
        #     expanded_mask = ATTN_MASK.unsqueeze(-1).to(tgt.device)  # (B, L, 1)
        #     add = add * expanded_mask  # (B, L, H) mask-aware addition

        mod = tgt + alpha * add
        return (mod,) + out[1:] if isinstance(out, tuple) else mod
        
    return module.register_forward_hook(_hook)




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
        def _hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out      # (B,L,H)
            h = h.detach().cpu()
            # if move_to_cpu:
            #     h = h.to("cpu", non_blocking=True)
            store[name].append(h.bfloat16())
            return out
        return _hook
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
        layers = [f"model.layers.{i}" for i in range(n)]
        if atten:
            layers = [f"model.layers.{i}.self_attn" for i in range(n)]
        print(f"Detected {n} layers: {layers}")

    # 2) GPT‑style: <top>.transformer.h
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
        layers =  [f"transformer.h.{i}" for i in range(n)]
        if atten:
            layers = [f"transformer.h.{i}.attn" for i in range(n)]
        print(f"Detected {n} layers: {layers}")

    # 3) Fallback – numeric names
    # else:
    #     n = getattr(model.config, "num_hidden_layers", None)
    #     if n is None:
    #         raise ValueError("Could not determine transformer block count.")
    #     layeres =  [f"layer_{i}" for i in range(n)]

    print(f"Detected {len(layers)} layers: {layers}")
    for n, m in model.named_modules():
        if (n.startswith("model.layers.") and n in layers):   # old typo variant
                  # GPT style
            print(f"Registering hook for {n}")
            handles.append(m.register_forward_hook(_factory(n)))
        elif n.startswith("transformer.h.") and n in layers:
            print(f"Registering hook for {n}")
            handles.append(m.register_forward_hook(_factory(n)))


    try:
        yield store
    finally:
        for h in handles:
            h.remove()