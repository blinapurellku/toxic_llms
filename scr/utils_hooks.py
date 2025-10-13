from ast import mod
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

torch.set_float32_matmul_precision("high")




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
    def _hook(_m, _inp, out):
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