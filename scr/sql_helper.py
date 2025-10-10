import json
import os
import re

import zstandard as zstd
from typing import Dict, List, Optional, Tuple, Union



def safe_name(s: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", s)

def save_prompts_responses(output_dir, model, dataset, layer_name, alpha, prompts, responses):
    safe_model = safe_name(model)
    folder = os.path.join(output_dir, safe_model)
    os.makedirs(folder, exist_ok=True)
    if dataset is not None:
        layer_name = f"{dataset}__{layer_name}"
    filename = f"{layer_name}__alpha_{alpha}.json.zst"
    path = os.path.join(folder, filename)

    obj = {
        "prompts": prompts,
        "responses": responses
    }
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")

    cctx = zstd.ZstdCompressor(level=9)
    with open(path, "wb") as f:
        f.write(cctx.compress(raw))

    print(f"Saved {len(prompts)} prompts/responses to {path}")

def load_prompts_responses(output_dir, model, dataset, layer_name, alpha):
    safe_model = safe_name(model)
    folder = os.path.join(output_dir, safe_model)
    if dataset is not None:
        layer_name = f"{dataset}__{layer_name}"
        
    filename = f"{layer_name}__alpha_{alpha}.json.zst"
    path = os.path.join(folder, filename)

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        data = dctx.decompress(f.read())
    obj = json.loads(data.decode("utf-8"))

    prompts = obj["prompts"]
    responses = obj["responses"]
    return prompts, responses




def _safe(s: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", s)

def _file_for_run(output_dir: str, model: str, layer: str, alpha: float | int, subdir="layers"):
    safe_model = _safe(model)
    folder = os.path.join(output_dir, safe_model, subdir)
    os.makedirs(folder, exist_ok=True)
    # tolerant name: alpha-1 or alpha-1.0 both fine
    alpha_part = f"{int(alpha)}" if isinstance(alpha, (int,)) or (isinstance(alpha, float) and alpha.is_integer()) else f"{alpha}"
    return os.path.join(folder, f"{layer}__alpha-{alpha_part}.json.zst")

def save_lists(path: str, prompts, responses, labels=None, level: int = 9):
    # keep it minimal; short keys = smaller files
    # p: prompts, r: responses, y: labels (e.g., classifier outputs)
    obj = {"p": list(prompts), "r": list(responses)}
    if labels is not None:
        obj["y"] = list(labels)
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    cctx = zstd.ZstdCompressor(level=level)
    with open(path, "wb") as f:
        f.write(cctx.compress(raw))

def load_lists(path: str):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        data = dctx.decompress(f.read())
    obj = json.loads(data.decode("utf-8"))
    prompts = obj["p"]
    responses = obj["r"]
    labels = obj.get("y", None)
    return prompts, responses, labels






''' the one below doesn't belong here but whatever '''

def _derive_layer_names(model) -> List[str]:
    """Return a list of *attribute paths* for each hidden‑state slot.

    The list length == ``num_hidden_layers + 1`` (extra slot 0 for embeddings).

    Examples
    --------
    * Llama‑family → ``[embeddings, 'model.model.layers.0', …]``
    * GPT‑2/GPT‑J   → ``[embeddings, 'transformer.h.0', …]``

    If the exact container list cannot be detected, we fall back to
    `'layer_{i}'` so the code still runs.
    """

    # 1) Common decoder‑only HF models: <top>.model.layers (Llama, Gemma, …)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
        return ["embeddings"] + [f"model.model.layers.{i}" for i in range(n)]

    # 2) GPT‑style: <top>.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
        return ["embeddings"] + [f"transformer.h.{i}" for i in range(n)]

    # 3) Fallback – numeric names
    n = getattr(model.config, "num_hidden_layers", None)
    if n is None:
        raise ValueError("Could not determine transformer block count.")
    return ["embeddings"] + [f"layer_{i}" for i in range(n)]