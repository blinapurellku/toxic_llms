import argparse
import os

# Disable TorchInductor as requested
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

# Optional: avoid error spam from Torch Dynamo
torch._dynamo.config.suppress_errors = False

# Define device globally
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_BASE_MODEL = False  # Set to True if you want to use the base model instead of the instruct model


def load_model_and_tokenizer(model_name, base_model=False):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer.pad_token = tokenizer.eos_token
    # also update the model config so generation doesn’t complain
    model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    return model, tokenizer


def generate_responses(model, tokenizer, prompts, base_model=False,
                       max_new_tokens=100, temperature=1.0, top_p=0.9, batch_size=4):
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i + batch_size]
        if base_model:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    # top_k=max_new_tokens,
                )
            for idx in range(len(batch_prompts)):
                response = tokenizer.decode(
                    output_ids[idx][inputs.input_ids.shape[1]:], skip_special_tokens=True
                )
                responses.append(response)
        else:
            batch_texts = []
            for prompt in batch_prompts:
                if hasattr(tokenizer, 'apply_chat_template'):
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True
                    )
                else:
                    text = f"<user>{prompt}</user>\n<assistant>"
                batch_texts.append(text)
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p
                )
            for idx in range(len(batch_prompts)):
                response = tokenizer.decode(
                    output_ids[idx][inputs.input_ids.shape[1]:], skip_special_tokens=True
                )
                responses.append(response)
    return responses


def evaluate_toxicity(texts, batch_size=4):
    # Returns list of dicts: {toxicity_score, classification_label}
    toxic_tokenizer = AutoTokenizer.from_pretrained(
        # "facebook/roberta-hate-speech-dynabench-r4-target"
        "cais/HarmBench-Llama-2-13b-cls"
    )
    toxic_model = AutoModelForSequenceClassification.from_pretrained(
        # "facebook/roberta-hate-speech-dynabench-r4-target"
        "cais/HarmBench-Llama-2-13b-cls"
    ).to(device)
    toxic_model.eval()
    results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating toxicity"):
            batch = texts[i:i + batch_size]
            inputs = toxic_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = toxic_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            tox_scores = probs[:, 1].cpu().tolist()
            labels = torch.argmax(probs, dim=1).cpu().tolist()
            for score, label in zip(tox_scores, labels):
                results.append({
                    "toxicity_score": score,
                    "classification_label": label
                })
    return results


def filter_fn(sample):
    tox = sample.get("prompt", {}).get("toxicity") if "prompt" in sample else None
    return tox is not None and tox > 0.3


def main():
    args = parse_args()
    args.base_model = USE_BASE_MODEL

    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)
    model.eval()

    print("Loading the real-toxicity-prompts dataset")
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    dataset = dataset.filter(filter_fn, batched=False)

    prompts = []
    for item in dataset:
        if len(prompts) >= args.num_prompts:
            break
        text = item.get("prompt", {}).get("text") or item.get("text")
        if text:
            prompts.append(text)

    responses = generate_responses(
        model, tokenizer, prompts, args.base_model,
        args.max_new_tokens, args.temperature, args.top_p, args.batch_size
    )

    toxicity_results = evaluate_toxicity(responses, batch_size=args.batch_size)

    # Build DataFrame with desired columns
    df = pd.DataFrame({
        "prompt": prompts,
        "model_output": responses,
        "toxicity_score": [r["toxicity_score"] for r in toxicity_results],
        "classification_label": [r["classification_label"] for r in toxicity_results]
    })

    os.makedirs(args.output_dir, exist_ok=True)
    model_short_name = args.model.split("/")[-1]
    output_file = os.path.join(args.output_dir, f"{model_short_name}_toxicity_results.csv")
    # df.to_csv(output_file, index=True, sep=";")

    print(f"Results saved to {output_file}")

    mean_toxicity = df['toxicity_score'].mean()
    total_samples = len(df)
    toxic_count = df['classification_label'].sum()
    ratio = toxic_count / total_samples if total_samples > 0 else 0
    print(f"Overall average toxicity score: {mean_toxicity:.4f}")
    print(f"Toxic classification count: {toxic_count}/{total_samples} ({ratio:.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate language models for toxicity")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it") #meta-llama/Llama-3.1-8B, google/gemma-2-2b-it, EleutherAI/gpt-neo-125m, Qwen/Qwen2-0.5B-Instruct, google/gemma-3-1b-it, google/gemma-3-1b-pt, google/gemma-2-2b")
    parser.add_argument("--base_model", action="store_true")
    parser.add_argument("--num_prompts", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="Master_thesis/results")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128) #256)
    return parser.parse_args()

if __name__ == "__main__":
    main()
