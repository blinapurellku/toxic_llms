import argparse
import gc
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

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


@torch.no_grad()
def classify_generation_datasets(
    prompts, responses, cls_model, cls_tokenizer, cls_model_id, behavior, templ= MISTRAL_CLS_PROMPT, batch_size: int = 8
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
    if isinstance(templ, dict): 
        templ = templ["prompt"]
        combined_inputs = [templ.format(behavior=u, generation=v) for u, v in zip(prompts, responses)]
        
    else:
        combined_inputs = [prompts[i] + responses[i] for i in range(len(prompts))]


    @find_executable_batch_size(starting_batch_size=batch_size)
    def _classify_batch_safely(bs):
        l_results = []
        for i in tqdm(
            range(0, len(combined_inputs), bs),
            desc=f"Classifying with {cls_model_id} (bs={bs})",
        ):
            batch_texts = combined_inputs[i : i + bs]
            enc = cls_tokenizer(batch_texts, return_tensors="pt", padding="longest").to(
                cls_model.device
            )
            with torch.inference_mode():
                out = cls_model.generate(
                    **enc, do_sample=False, max_new_tokens=1
                ).cpu()

            toks = out[:, enc.input_ids.shape[1] :]
            decoded = cls_tokenizer.batch_decode(toks, skip_special_tokens=True)

            for d in decoded:
                label = d.strip().lower()
                if label == "yes":
                    is_toxic = 1
                elif label == "no":
                    is_toxic = 0
                else:
                    is_toxic = -1
                    print(f"Unexpected label '{label}' in: {d}")

                l_results.append(is_toxic)
        print(len(l_results), "responses classified")
        
        del enc, out
        if torch.cuda.is_available():
            gc.collect()               
            torch.cuda.empty_cache()
        return l_results

    results = _classify_batch_safely()

    return results



@torch.no_grad()
def classify_generation_truthfulqa(
    prompts, responses, cls_model, cls_tokenizer, cls_model_id, behavior, templ, batch_size: int = 8
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
    combined_inputs = [
        templ.format(question=u, answer=v) for u, v in zip(prompts, responses)
    ]

    @find_executable_batch_size(starting_batch_size=batch_size)
    def _classify_batch_safely(bs):
        l_results = []
        for i in tqdm(
            range(0, len(combined_inputs), bs),
            desc=f"Classifying with {cls_model_id} (bs={bs})", mininterval=10
        ):
            batch_texts = combined_inputs[i : i + bs]
            enc = cls_tokenizer(batch_texts, return_tensors="pt", padding="longest").to(
                cls_model.device
            )
            with torch.inference_mode():
                out = cls_model.generate(
                    **enc, do_sample=False, max_new_tokens=1
                ).cpu()

            toks = out[:, enc.input_ids.shape[1] :]
            decoded = cls_tokenizer.batch_decode(toks, skip_special_tokens=True)

            for d in decoded:
                label = d.strip().lower()
                if label == "no":
                    is_toxic = 1
                elif label == "yes":
                    is_toxic = 0
                else:
                    is_toxic = -1
                    print(f"Unexpected label '{label}' in: {d}")

                l_results.append(is_toxic)
        # print(len(l_results), "responses classified")
        
        del enc, out
        if torch.cuda.is_available():
            gc.collect()               
            torch.cuda.empty_cache()
        return l_results

    results = _classify_batch_safely()

    return results


@torch.no_grad()
def classify_generation_classification(
    prompts, responses, cls_model, cls_tokenizer, cls_model_id, behavior, templ=None, batch_size: int = 8
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
    @find_executable_batch_size(starting_batch_size=batch_size)
    def _classify_batch_safely(bs):
        l_results = []
        for i in tqdm(
            range(0, len(responses), bs),
            desc=f"Classifying with {cls_model_id} (bs={bs})", mininterval=10
        ):
            batch_texts = responses[i : i + bs]
            enc = cls_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, #max_length=512)
                ).to(cls_model.device)
            
            with torch.inference_mode():
                out = cls_model(**enc).logits
            
            decoded = out.argmax(-1).tolist()
            
            for is_toxic in decoded:             
                l_results.append(is_toxic)
        
        del enc, out
        if torch.cuda.is_available():
            gc.collect()               
            torch.cuda.empty_cache()
        return l_results

    results = _classify_batch_safely()

    return results


@torch.no_grad()
def classify_generation_bias(
    prompts, responses, cls_model, cls_tokenizer, cls_model_id, behavior, templ=None, batch_size: int = 8
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = []
    for r in responses:
        txt = r.lower()
        r = cls_model(txt)
        results.append(r)

    return results
    





classification_dict = {
    'cais/HarmBench-Mistral-7b-val-cls': classify_generation_datasets,
    'classifier_function': classify_generation_bias,
    'allenai/truthfulqa-truth-judge-llama2-7B': classify_generation_truthfulqa,
    'facebook/roberta-hate-speech-dynabench-r4-target': classify_generation_classification,

}


def classify_generation(
    prompts, responses, cls_model, cls_tokenizer, cls_model_id, behavior, templ, batch_size: int = 8
):
    if cls_model_id not in classification_dict:
        raise ValueError(f"Classifier model {cls_model_id} not recognized. Available models: {list(classification_dict.keys())}")
    
    classify_fn = classification_dict[cls_model_id]
    results = classify_fn(
        prompts, responses, cls_model, cls_tokenizer, cls_model_id, behavior, templ, batch_size
    )
    return results