import gc
import os
from typing import List, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

torch.set_float32_matmul_precision("high")


from accelerate.utils import find_executable_batch_size
from tqdm import tqdm

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





@torch.no_grad()
def run_prompting(
    model,
    tokenizer,
    prompts,
    base_model: bool = False,
    template: dict | None = None,
    starting_batch_size: int = 64,
    atten: bool = False,
    aggregate: str = None,
    tmp_dir: str = "./tmp",
):
    """Generate logits **and** hidden states for *prompts* with auto‑batch‑size."""

    run_kwargs = {
        "pad_token_id": tokenizer.pad_token_id,
        # "output_hidden_states": True,
    }
    # layer_names = _derive_layer_names(model)[1:]
    with capture_all_layers(model, move_to_cpu=True, atten=atten) as acts:
        @find_executable_batch_size(starting_batch_size=starting_batch_size)
        def _inner(bs):
            all_logits, all_masks = [], []
            all_hidden = defaultdict(list)  # Store hidden states    
            all_hidden_sum = defaultdict(list)  # Store hidden states
            id_ = 0
            for i in tqdm(range(0, len(prompts), bs), desc=f"Generating (bs={bs})"):
                
                chunk = prompts[i : i + bs]
                if base_model:
                    wrapped = chunk
                else:
                    if template is None:
                        raise ValueError(
                            "A chat template must be supplied when base_model=False"
                        )
                    wrapped = [template["prompt"].format(instruction=p) for p in chunk]

                # enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
                enc = tokenizer(
                    wrapped, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)
                # model.config.num_attention_heads
                with torch.inference_mode():
                    out = model(**enc, **run_kwargs)
                    # print(acts)
                for layer, tensors in acts.items():
                    h_state = tensors[0].cpu() * enc.attention_mask.unsqueeze(-1).cpu()   # (B, L, 1)
                    # print(tensors[0].shape)
                    if atten: 
                        H = model.config.num_attention_heads
                        B, L, _ = h_state.shape
                        h_state = h_state.view(B, L, H, -1).permute(0, 2, 1, 3)  # (B, H, L, HD)

                        token_counts = enc.attention_mask.cpu().sum(dim=1).clamp(min=1)
                        token_counts = token_counts.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
                        h_states = h_state.sum(dim=2) / token_counts  # (B, H, HD)
                        all_hidden_sum[layer].append(h_states)

                        h_states = h_state[:, :, -1, :]
                        print(layer, h_state.shape)
                        all_hidden[layer].append(h_states)
                    else:
                        token_counts = enc.attention_mask.cpu().sum(dim=1).clamp(min=1)  # (B, 1), to prevent divide-by-zero
                        token_counts = token_counts.unsqueeze(1)
                        h_states = h_state.sum(dim=1) / token_counts  # (B, HD)
                        all_hidden_sum[layer].append(h_states)


                        h_states = h_state[:, -1, :]   # (B, HD)
                        all_hidden[layer].append(h_states)

                    # token_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)    # (B, 1)
                    # print(h_state.shape)
                    # seq_avg = (tensors[0].cpu() * mask.cpu()).sum(dim=1) / token_counts.cpu()
                    # all_hidden[layer].append(h_state)

                    del tensors[:], h_state, h_states #, token_counts
                    gc.collect()
                    torch.cuda.empty_cache()

                id_ += 1
                all_logits.append(out.logits[:, -1, :].cpu())
                all_masks.append(enc.attention_mask.cpu())

                print(f"Generated {len(chunk)} responses.")


            return all_logits, all_masks, all_hidden, all_hidden_sum

        all_logits, all_masks, all_hidden, all_hidden_sum = _inner()

    # L_max = max(t.size(1) for t in all_masks)
    # logits = torch.cat([F.pad(t, (0, 0, 0, L_max - t.size(1)))
    #                     for t in all_logits], dim=0)
    # masks  = torch.cat([F.pad(t, (0, L_max - t.size(1)))
    #                     for t in all_masks], dim=0)
    max_len = max(m.shape[1] for m in all_masks)

    

    # pad masks on the left of the seq dimension
    padded_masks = [
        F.pad(mask, (max_len - mask.size(1), 0))
        for mask in all_masks
    ]

    # padded_states = {}
    # for layer, states in acts.items():
    #         padded_states[layer] = [
    #             F.pad(h, (0, 0, max_len - h.size(1), 0))
    #             for h in states
            # ]
    all_logits = torch.cat(all_logits, dim=0)       # [total_examples, max_len, vocab]
    padded_masks  = torch.cat(padded_masks,  dim=0)       # [total_examples, max_len]
    # states_tensor = {
    #     layer: torch.cat(h_list, dim=0)                   # [total_examples, max_len, hid_dim]
    #     for layer, h_list in padded_states.items()
    # }  
    all_hidden = {layer: torch.cat(h_list, dim=0) for layer, h_list in all_hidden.items()}

    all_hidden_sum = {layer: torch.cat(h_list, dim=0) for layer, h_list in all_hidden_sum.items()}
    
    print(all_hidden[list(all_hidden.keys())[0]].shape)
    return all_logits, padded_masks, all_hidden, all_hidden_sum





    

@torch.no_grad()
def run_prompting_generate(
    model,
    tokenizer,
    prompts,
    responses: Optional[List[str]] = None,
    base_model: bool = False,
    starting_batch_size: int = 4,
    template: dict | None = None,
    atten: bool = False,
    aggregate: str = None, # or sum
):
    """Generate logits **and** hidden states for *prompts* with auto‑batch‑size."""
    run_kwargs = {
        # "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        # "return_dict_in_generate": True,  # Return a more detailed output object
    }
    
    if responses is not None:
        prompts = [p + r for p, r in zip(prompts, responses)]
        print("here")

    with capture_all_layers(model, move_to_cpu=True, atten=atten) as acts:
        @find_executable_batch_size(starting_batch_size=starting_batch_size)
        def _inner(bs):
            all_hidden = defaultdict(list)  # Store hidden states    
            all_hidden_sum = defaultdict(list)  # Store hidden states
            all_hidden_last = defaultdict(list)  # Store hidden states
            for i in tqdm(range(0, len(prompts), bs), desc=f"Generating (bs={bs})"):
                chunk = prompts[i : i + bs]
                chunk_f = responses[i : i + bs] if responses else chunk

                if base_model:
                    wrapped = chunk
                    wrapped_f = chunk_f 
                else:
                    if template is None:
                        raise ValueError(
                            "A chat template must be supplied when base_model=False"
                        )
                    wrapped = [template["prompt"].format(instruction=p) for p in chunk]

                    wrapped_f = [template["prompt"].format(instruction=p) for p in chunk_f]

                # enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
                enc = tokenizer(
                    wrapped, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)

                enc_f = tokenizer(wrapped_f, return_tensors="pt", padding=True, truncation=True
                )#.to(model.device)
                try:
                    with torch.inference_mode():
                        out = model(**enc, **run_kwargs)
                    for layer, tensors in acts.items():
                        h_state = tensors[0].cpu() * enc.attention_mask.unsqueeze(-1).cpu()   # (B, L, 1)
                        if atten: 
                            H = model.config.num_attention_heads
                            B, L, _ = h_state.shape
                            h_state = h_state.view(B, L, H, -1).permute(0, 2, 1, 3)  # (B, H, L, HD)

                            if aggregate == "sum":
                                token_counts = enc.attention_mask.cpu().sum(dim=1).clamp(min=1)
                                token_counts = token_counts.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
                                h_state = h_state.sum(dim=2) / token_counts  # (B, H, HD)
                            else:
                                h_state = h_state[:, :, -1, :]
                                print(layer, h_state.shape)
                        else:
                            
                            token_counts = enc.attention_mask.cpu().sum(dim=1).clamp(min=1)  # (B, 1), to prevent divide-by-zero
                            token_counts = token_counts.unsqueeze(1)
                            h_states = h_state.sum(dim=1) / token_counts  # (B, HD)
                            all_hidden_sum[layer].append(h_states)
                        

                            h_states = h_state[:, -1, :]   # (B, HD)
                            all_hidden_last[layer].append(h_states)
                        

                            get_f = enc_f.attention_mask.sum(dim=1).clamp(min=1)  # (B, 1), to prevent divide-by-zero
                            l = enc.attention_mask.shape[1] #.clamp(min=1)  # (B, 1), to prevent divide-by-zero
                            use = []
                            for i in range(len(get_f)):
                                use.append(h_state[i, l-get_f[i], :])
                            h_states = torch.stack(use, dim=0) # (B, HD)
                            all_hidden[layer].append(h_states)

                        # all_hidden[layer].append(h_state)

                        del tensors[:], h_state, h_states #, token_counts
                        gc.collect()
                        torch.cuda.empty_cache()

                finally:
                    # Hard cleanup so bs retries / next batches don't see stale captures
                    for lst in acts.values():
                        if lst:
                            del lst[:]
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  

                print(f"Generated {len(chunk)} responses.")
           
                


            return all_hidden_sum, all_hidden_last, all_hidden

        all_hidden_sum, all_hidden_last, all_hidden = _inner()

    all_hidden_sum = {layer: torch.cat(h_list, dim=0) for layer, h_list in all_hidden_sum.items()}
    print(all_hidden_sum[list(all_hidden_sum.keys())[0]].shape)

    all_hidden_last = {layer: torch.cat(h_list, dim=0) for layer, h_list in all_hidden_last.items()}
    print(all_hidden_last[list(all_hidden_last.keys())[0]].shape)

    all_hidden = {layer: torch.cat(h_list, dim=0) for layer, h_list in all_hidden.items()}
    print(all_hidden[list(all_hidden.keys())[0]].shape)

    return all_hidden_sum, all_hidden_last, all_hidden


