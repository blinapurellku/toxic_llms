import argparse
import datetime
import gc
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union

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
from templates import LLAMA_CLS_PROMPT, get_template, MISTRAL_CLS_PROMPT
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



@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts,
    base_model: bool = False,
    max_new_tokens: int = 100,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    starting_batch_size: int = 4,
    template: dict | None = None,
    output_dir: str = "./",
):
    """Generate *responses* for `prompts`, guaranteeing a chat‑template wrap
    (unless `base_model=True`) and auto‑adapt batch size to GPU capacity."""

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        # "return_dict_in_generate": True,  # Return a more detailed output object
    }
    if do_sample:
        gen_kwargs.update(
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
        )
    
    

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def _inner(bs):
        responses = [] 
        for i in tqdm(range(0, len(prompts), bs), desc=f"Generating (bs={bs})", mininterval=10):
            chunk = prompts[i : i + bs]
            if base_model:
                wrapped = chunk
            else:
                if template is None:
                    raise ValueError(
                        "A chat template must be supplied when base_model=False"
                    )
                wrapped = [template["prompt"].format(instruction=p) for p in chunk]

            enc = tokenizer(
                wrapped, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.inference_mode():
                generation_output = model.generate(**enc, **gen_kwargs).cpu()
                
            # With return_dict_in_generate=True, we get a more detailed output object
            # sequences = generation_output.sequences
            
            for j in range(len(chunk)):
                # ids = sequences[j]  # [seq_len]

                decoded = tokenizer.decode(generation_output[j][enc.input_ids.shape[1] :], skip_special_tokens=True).strip()

                if not decoded:
                    print(f" Empty generation retrying for: {chunk[j]}")

                    with torch.inference_mode():
                        retry_out = model.generate(
                            input_ids=enc.input_ids[j].unsqueeze(0),
                            attention_mask=enc.attention_mask[j].unsqueeze(0),
                            **gen_kwargs,
                        ).cpu()

                    decoded = tokenizer.decode(
                        retry_out[0][enc.input_ids.shape[1] :], skip_special_tokens=True
                    ).strip()

                    
                responses.append(decoded)

            # del enc, generation_output # Free memory
            # if torch.cuda.is_available():
            #     gc.collect()
            #     torch.cuda.empty_cache()

        return responses
    
    responses = _inner()
    return responses





