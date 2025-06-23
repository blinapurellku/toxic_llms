import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

# --- Environment cleanup ---
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Get toxicity labels via sequence classifier ---
def get_toxicity_labels(texts, batch_size=8):
    classifier = AutoModelForSequenceClassification.from_pretrained(
        "facebook/roberta-hate-speech-dynabench-r4-target"
    ).to(device)
    tok = AutoTokenizer.from_pretrained(
        "facebook/roberta-hate-speech-dynabench-r4-target"
    )
    classifier.eval()
    labels = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = classifier(**inputs)
            probs = torch.softmax(out.logits, dim=-1)
            labels.extend(torch.argmax(probs, dim=-1).cpu().tolist())
    return labels

# --- Hook to capture residual activations ---
class ActivationHook:
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.last_activation = None
    def hook_fn(self, module, inp, out):
        # out shape [batch, seq, dim]
        self.last_activation = out.detach().cpu()
    def remove(self):
        self.handle.remove()

# --- Main routine ---
def run(args):
    # load causal LM
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()

    # load prompts
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    texts = [itm["prompt"]["text"] for itm in ds if "prompt" in itm and "text" in itm["prompt"]][:args.num_samples]

    # get toxicity labels
    labels = np.array(get_toxicity_labels(texts))  # 1 to toxic, 0 otherwise

    # prepare hooks to capture residuals
    # for GPT-like: model.transformer.h
    blocks = model.model.decoder.layers if hasattr(model.model, 'decoder') else model.transformer.h
    hooks = [ActivationHook(block) for block in blocks]

    # collect activations
    activations = []  # [num_samples, num_layers, dim]
    with torch.no_grad():
        for txt in tqdm(texts, desc="Collecting activations"):
            inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True).to(device)
            _ = model.generate(**inputs, max_new_tokens=0)
            # pool each layer's activation
            layer_feats = [hook.last_activation.mean(dim=[0,1]).numpy() for hook in hooks]
            activations.append(layer_feats)
    # remove capture hooks
    for h in hooks: h.remove()

    activations = np.array(activations)  # shape (N, L, D)
    y = labels

    # compute toxicity direction per layer
    tox_mean = activations[y==1].mean(axis=0)       # (L, D)
    nontox_mean = activations[y==0].mean(axis=0)    # (L, D)
    directions = tox_mean - nontox_mean             # (L, D)

    # save data & directions
    np.savez(args.output_path, activations=activations, labels=y, directions=directions)

    # train linear classifier per layer
    print("Training linear classifiers per layer:")
    for layer in range(directions.shape[0]):
        X = activations[:, layer, :]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_test = accuracy_score(y_test, clf.predict(X_test))
        print(f"Layer {layer}: Train Acc={acc_train:.4f}, Test Acc={acc_test:.4f}")

    # perturb non-toxic examples by adding direction vectors
    if args.perturb:
        print("Generating perturbed outputs for non-toxic samples...")
        # convert directions to torch tensors
        dirs = torch.tensor(directions, device=device)
        # register modification hooks
        handles_mod = []
        for layer, block in enumerate(blocks):
            # hook that adds direction to the residual output
            def make_hook(dir_vec):
                def hook_fn(module, inp, out):
                    # out shape [batch, seq, dim]
                    return out + dir_vec[None, None, :]
                return hook_fn
            handles_mod.append(block.register_forward_hook(make_hook(dirs[layer])))

        # generate perturbed completions
        for i, txt in enumerate(texts[:args.perturb_count]):
            if y[i] == 0:
                inputs = tokenizer(txt, return_tensors="pt").to(device)
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                original = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                print(f"Sample {i} original: {txt}")
                print(f"Sample {i} perturbed: {original}\n")

        # remove modification hooks
        for h in handles_mod: h.remove()

    print("Done.")

# --- Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations and analyze toxicity directions")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_path", type=str, default="activation_data.npz")
    parser.add_argument("--perturb", action="store_true", help="Apply directional perturbation to non-toxic samples")
    parser.add_argument("--perturb_count", type=int, default=5, help="Number of non-toxic samples to perturb")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    run(parse_args())


import torch.nn as nn


def inject_identity_layers(model):
    # this assumes your model has a .transformer.h list of blocks
    for i, block in enumerate(model.transformer.h):
        # attach an identity module
        block.identity = nn.Identity()
        # stash the old forward
        original_forward = block.forward
        # define a new forward that applies the identity to the post‐residual output
        def patched_forward(self, x, **kwargs):
            # run the original block (attention + MLP + residual adds inside)
            out = original_forward(x, **kwargs)
            # now pass that result through the identity
            return self.identity(out)
        # bind it
        block.forward = patched_forward.__get__(block, block.__class__)

# inject_identity_layers(model)


############ TOOO GET THE TOXICITY OF GENERATED OUTPUTS ############




import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

# --- Environment cleanup ---
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Get toxicity labels via sequence classifier ---
def get_toxicity_labels(texts, batch_size=8):
    classifier = AutoModelForSequenceClassification.from_pretrained(
        "facebook/roberta-hate-speech-dynabench-r4-target"
    ).to(device)
    tok = AutoTokenizer.from_pretrained(
        "facebook/roberta-hate-speech-dynabench-r4-target"
    )
    classifier.eval()
    labels = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = classifier(**inputs)
            probs = torch.softmax(out.logits, dim=-1)
            labels.extend(torch.argmax(probs, dim=-1).cpu().tolist())
    return labels  # 1 = toxic, 0 = non‐toxic

# --- Hook to capture residual activations ---
class ActivationHook:
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.last_activation = None
    def hook_fn(self, module, inp, out):
        # out shape [batch, seq, dim]
        self.last_activation = out.detach().cpu()
    def remove(self):
        self.handle.remove()

# --- Main routine ---
def run(args):
    # load causal LM
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()

    # load prompts
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    texts = [itm["prompt"]["text"] for itm in ds if "prompt" in itm and "text" in itm["prompt"]][:args.num_samples]

    # 1) Generate completions
    print("Generating completions…")
    completions = []
    with torch.no_grad():
        for txt in tqdm(texts, desc="Generating"):
            inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True).to(device)
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            completions.append(gen_text)

    # 2) Classify toxicity of generated texts
    print("Classifying generated texts for toxicity…")
    gen_labels = np.array(get_toxicity_labels(completions))
    tox_rate = gen_labels.mean()
    print(f"Overall toxicity rate in generated texts: {tox_rate:.3%}")

    # (Optional) If you still want to compare prompt vs generation:
    # prompt_labels = np.array(get_toxicity_labels(texts))
    # print(f"Prompts toxic: {prompt_labels.mean():.3%}")
    # print(f"Generations toxic: {tox_rate:.3%}")

    # 3) (Unchanged) Hook in and collect activations based on prompts
    blocks = model.model.decoder.layers if hasattr(model.model, 'decoder') else model.transformer.h
    hooks = [ActivationHook(block) for block in blocks]
    activations = []
    with torch.no_grad():
        for txt in tqdm(texts, desc="Collecting activations"):
            inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True).to(device)
            # still use max_new_tokens=0 here to only capture prompt activations
            _ = model.generate(**inputs, max_new_tokens=0)
            layer_feats = [hook.last_activation.mean(dim=[0,1]).numpy() for hook in hooks]
            activations.append(layer_feats)
    for h in hooks: h.remove()
    activations = np.array(activations)  # shape (N, L, D)

    # 4) Compute toxicity direction per layer (using generation labels!)
    tox_mean = activations[gen_labels==1].mean(axis=0)
    nontox_mean = activations[gen_labels==0].mean(axis=0)
    directions = tox_mean - nontox_mean
    np.savez(args.output_path, activations=activations, labels=gen_labels, directions=directions)

    # 5) Train linear classifiers per layer
    print("Training linear classifiers per layer:")
    for layer in range(directions.shape[0]):
        X = activations[:, layer, :]
        X_train, X_test, y_train, y_test = train_test_split(X, gen_labels, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_test = accuracy_score(y_test, clf.predict(X_test))
        print(f"Layer {layer}: Train Acc={acc_train:.4f}, Test Acc={acc_test:.4f}")

    # 6) Perturb non‐toxic samples if requested
    if args.perturb:
        print("Generating perturbed outputs for non-toxic samples…")
        dirs = torch.tensor(directions, device=device)
        handles_mod = []
        for layer, block in enumerate(blocks):
            def make_hook(dir_vec):
                def hook_fn(module, inp, out):
                    return out + dir_vec[None, None, :]
                return hook_fn
            handles_mod.append(block.register_forward_hook(make_hook(dirs[layer])))

        for i, (txt, lbl) in enumerate(zip(texts, gen_labels[:len(texts)])):
            if lbl == 0 and i < args.perturb_count:
                inputs = tokenizer(txt, return_tensors="pt").to(device)
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                perturbed = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                print(f"Sample {i} original prompt: {txt}")
                print(f"Sample {i} perturbed generation: {perturbed}\n")

        for h in handles_mod: h.remove()

    print("Done.")

# --- Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations and analyze toxicity directions")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_path", type=str, default="activation_data.npz")
    parser.add_argument("--perturb", action="store_true", help="Apply directional perturbation to non-toxic samples")
    parser.add_argument("--perturb_count", type=int, default=5, help="Number of non-toxic samples to perturb")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    run(parse_args())
