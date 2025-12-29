
import argparse
from typing import List, Optional, Callable, Sequence, Any
from collections import defaultdict
import json
import os
import re
import gc

import torch
from utils_load_dataset_and_models import load_dataset, load_model_and_tokenizer

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

torch.set_float32_matmul_precision("high")


import numpy as np
from accelerate.utils import find_executable_batch_size
from utils_templates import get_template
from transformers import (BitsAndBytesConfig)
import math
from utils_hooks import steering_vector_hook_adaptive as steering_vector_hook
from utils_hooks import PerBatchAlpha

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


@torch.no_grad()
def perplexity_prompts(model, tokenizer, prompts, template, base_model, starting_bs=32, per_sample_alphas: Optional[Sequence[float]] = None,
    set_alpha_fn: Optional[Callable[[Sequence[float]], None]] = None,):
    # model.eval() - is in eval mode
    # total_nll, total_tokens = 0.0, 0
    # Basic validation for alphas length, if provided
    if per_sample_alphas is not None and len(per_sample_alphas) != len(prompts):
        raise ValueError(
            f"`per_sample_alphas` length {len(per_sample_alphas)} != number of prompts {len(prompts)}"
        )
    
    @find_executable_batch_size(starting_batch_size=starting_bs)
    def _ppl_batch(batch_size):
        # nonlocal total_nll, total_tokens
        total_nll, total_tokens = 0.0, 0
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            if set_alpha_fn is not None and per_sample_alphas is not None:
                batch_alphas = per_sample_alphas[i : i + len(chunk)]
                set_alpha_fn(batch_alphas)

            if base_model:
                wrapped = chunk
            else:
                if template is None:
                    raise ValueError(
                        "A chat template must be supplied when base_model=False"
                    )
                wrapped = [template["prompt"].format(instruction=p) for p in chunk]

            enc = tokenizer(wrapped, return_tensors="pt", padding=True, truncation=True).to(model.device)
            labels = enc.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100  # ignore pads
            with torch.inference_mode():
                out = model(**enc, labels=labels)

            valid_tokens = (labels != -100).sum().item()
            nll = out.loss.item() * valid_tokens
            total_nll += nll
            total_tokens += valid_tokens

            # nll = out.loss.item() * enc.input_ids.numel()
            # total_nll += nll
            # total_tokens += enc.input_ids.numel()

            del enc, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return total_nll, total_tokens

    total_nll, total_tokens = _ppl_batch()
    return math.exp(total_nll / total_tokens)

   





@torch.no_grad()
def perplexity_generated(model, tokenizer, prompts, generations, starting_bs=32, device="cuda"):
    assert len(prompts) == len(generations), "Mismatch: prompts and generations must be same length"
    model.eval()
    total_nll, total_tokens = 0.0, 0

    @find_executable_batch_size(starting_batch_size=starting_bs)
    def _ppl_batch(batch_size):
        # nonlocal total_nll, total_tokens
        total_nll, total_tokens = 0.0, 0
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_gens = generations[i : i + batch_size]

            # full text (prompt+gen)
            enc_full = tokenizer(
                [p + g for p, g in zip(batch_prompts, batch_gens)],
                return_tensors="pt", padding=True, truncation=True
            ).to(device)

            # prompt lengths
            enc_prompts = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_lens = (enc_prompts.input_ids != tokenizer.pad_token_id).sum(dim=1).cpu()

            # labels: mask out prompt tokens
            labels = enc_full.input_ids.clone()
            for j, L in enumerate(prompt_lens):
                labels[j, :L] = -100

            with torch.inference_mode():
                out = model(input_ids=enc_full.input_ids,
                            attention_mask=enc_full.attention_mask,
                            labels=labels)

            valid_tokens = (labels != -100).sum().item()
            nll = out.loss.item() * valid_tokens
            total_nll += nll
            total_tokens += valid_tokens

            del enc_full, enc_prompts, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return total_nll, total_tokens

    total_nll, total_tokens = _ppl_batch()
    return math.exp(total_nll / total_tokens)