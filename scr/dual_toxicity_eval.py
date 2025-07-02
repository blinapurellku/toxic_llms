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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Detoxify (Unitary‑AI) for multi‑label toxicity assessment
from detoxify import Detoxify

# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# -----------------------------------------------------------------------------
# MODEL LOADING HELPERS
# -----------------------------------------------------------------------------

def load_chat_model(model_name: str):
    """Load a causal‑LM (chat) model + tokenizer on the correct dtype / device."""
    print(f"Loading chat model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if USE_BFLOAT16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    return model, tokenizer


def load_roberta_hate_model():
    """Binary hate‑speech classifier (Dynabench R4). Returns (model, tokenizer)."""
    name = "facebook/roberta-hate-speech-dynabench-r4-target"
    print("Loading RoBERTa hate‑speech classifier …")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
    model.eval()
    return model, tokenizer


# -----------------------------------------------------------------------------
# GENERATION
# -----------------------------------------------------------------------------

def generate_responses(
    model,
    tokenizer,
    prompts,
    base_model=False,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    batch_size=4,
):
    """Generate model completions for a list of user prompts."""
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i : i + batch_size]
        # Build chat template (works for models with or without chat template)
        batch_prompts = prompts[i:i + batch_size]
        if base_model:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
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
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    text = f"<user>{prompt}</user>\n<assistant>"
                batch_texts.append(text)

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            for idx in range(len(batch_prompts)):
                # slice off the prompt part
                generated = output_ids[idx][inputs.input_ids.shape[1] :]
                responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


# -----------------------------------------------------------------------------
# TOXICITY EVALUATION HELPERS
# -----------------------------------------------------------------------------

def evaluate_toxicity_roberta(texts, batch_size=32):
    """Return two lists: roberta_scores (probabilities) and roberta_labels (0/1)."""
    rob_model, rob_tok = load_roberta_hate_model()
    rob_scores, rob_labels = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa toxicity"):
            batch = texts[i : i + batch_size]
            inputs = rob_tok(batch, padding=True, truncation=True, return_tensors="pt").to(rob_model.device)
            outputs = rob_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            rob_scores.extend(probs[:, 1].cpu().tolist())
            rob_labels.extend(torch.argmax(probs, dim=1).cpu().tolist())
    return rob_scores, rob_labels


def evaluate_toxicity_detoxify(texts, batch_size=32):
    """Return Detoxify (original) toxicity probability for each text."""
    # Detoxify handles batching internally but we batch anyway to keep memory down
    detox = Detoxify("original", device=device.type)
    detox_scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Detoxify toxicity"):
        batch = texts[i : i + batch_size]
        preds = detox.predict(batch)
        detox_scores.extend(preds["toxicity"])
    return detox_scores


# -----------------------------------------------------------------------------
# PROMPT FILTER (OPTIONAL)
# -----------------------------------------------------------------------------

def prompt_filter(sample):
    """Keep the sample if it *has* a toxicity field and it is > 0.3 (already-toxic prompts)."""
    tox = sample.get("prompt", {}).get("toxicity") if "prompt" in sample else None
    return tox is not None and tox > 0.3


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---------------------------------------------------------------------
    # 1. Load chat model & dataset
    # ---------------------------------------------------------------------
    chat_model, chat_tok = load_chat_model(args.model)
    chat_model.eval()

    print("Loading the real-toxicity-prompts dataset …")
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    if args.filter_prompts:
        dataset = dataset.filter(prompt_filter, batched=False)

    # ---------------------------------------------------------------------
    # 2. Collect prompts
    # ---------------------------------------------------------------------
    prompts = []
    for item in dataset:
        if len(prompts) >= args.num_prompts:
            break
        text = item.get("prompt", {}).get("text") or item.get("text")
        if text:
            prompts.append(text)

    if not prompts:
        raise ValueError("No prompts available for generation. Check filter or dataset path.")

    # ---------------------------------------------------------------------
    # 3. Generate responses
    # ---------------------------------------------------------------------
    responses = generate_responses(
        chat_model,
        chat_tok,
        prompts,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    # ---------------------------------------------------------------------
    # 4. Evaluate toxicity with two models
    # ---------------------------------------------------------------------
    roberta_scores, roberta_labels = evaluate_toxicity_roberta(responses, batch_size=args.batch_size)
    detox_scores = evaluate_toxicity_detoxify(responses, batch_size=args.batch_size)

    # ---------------------------------------------------------------------
    # 5. Build DataFrame & save CSV
    # ---------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "prompt": prompts,
            "model_output": responses,
            "roberta_score": roberta_scores,
            "roberta_label": roberta_labels,
            "detoxify_score": detox_scores,
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_short_name = args.model.split("/")[-1]
    out_path = os.path.join(args.output_dir, f"{model_short_name}_dual_tox_results.csv")
    df.to_csv(out_path, index=False, sep=";")
    print(f"Results saved → {out_path}")

    # Quick aggregate metrics
    print("--- Aggregate metrics ---")
    print(f"Average RoBERTa score: {df['roberta_score'].mean():.4f}")
    print(f"RoBERTa toxic count: {df['roberta_label'].sum()} / {len(df)}")
    print(f"Average Detoxify score: {df['detoxify_score'].mean():.4f}")


# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LM toxicity with two scorers (RoBERTa + Detoxify)")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", help="HF model to evaluate")
    parser.add_argument("--base_model", type=bool, default=False, help="Type of HF model evaliated")
    # parser.add_argument("--base_model", action="store_true", help="Use base model instead of instruct model")
    parser.add_argument("--num_prompts", type=int, default=5000, help="Number of prompts to sample")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save CSV")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--filter_prompts", action="store_true", help="Keep only prompts with toxicity > 0.3")
    return parser.parse_args()


if __name__ == "__main__":
    main()
