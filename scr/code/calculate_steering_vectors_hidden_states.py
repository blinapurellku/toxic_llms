#!/usr/bin/env python3
"""
Calculate steering vectors from hidden states for toxicity reduction.
This script loads the hidden states activations and computes steering vectors.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_activations_and_labels(activations_dir, labels_file, activation_type="hidden_states"):
    """
    Load hidden states activations and corresponding toxicity labels.
    
    Args:
        activations_dir: Directory containing activation files
        labels_file: CSV file with prompt IDs and toxicity labels
        activation_type: Type of activations to load (hidden_states or residual_stream)
        
    Returns:
        all_activations: Dictionary of activations by layer, each containing a list of tensors
        labels: Dictionary mapping prompt IDs to binary toxicity labels (1=toxic, 0=non-toxic)
    """
    print(f"Loading {activation_type} and toxicity labels...")
    
    # Load labels
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    labels_df = pd.read_csv(labels_file)
    
    # Create a dictionary mapping prompt IDs to binary labels
    labels = {}
    for _, row in labels_df.iterrows():
        prompt_id = row["prompt_id"]
        is_toxic = 1 if row["is_toxic"] else 0
        labels[prompt_id] = is_toxic
    
    print(f"Loaded {len(labels)} labels, {sum(labels.values())} toxic, {len(labels) - sum(labels.values())} non-toxic")
    
    # Check if activations directory exists
    if not os.path.exists(activations_dir):
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    # Initialize dictionary to store activations by layer
    all_activations = {}
    
    # First, determine available layers and prompts
    all_files = os.listdir(activations_dir)
    activation_files = [f for f in all_files if f.startswith(f"{activation_type}_") and f.endswith(".safetensors")]
    
    if not activation_files:
        raise ValueError(f"No activation files found in {activations_dir} for type {activation_type}")
    
    # Extract unique layers from filenames
    layers = set()
    for file_name in activation_files:
        # Extract layer from filename (format: hidden_states_layer_X_promptY.safetensors)
        parts = file_name.split("_")
        if len(parts) >= 3 and parts[2] == "layer":
            layer_num = int(parts[3])
            layers.add(layer_num)
    
    layers = sorted(layers)
    print(f"Found activations for {len(layers)} layers: {layers}")
    
    # Initialize the activations dictionary with empty lists for each layer
    for layer in layers:
        all_activations[layer] = {"toxic": [], "non-toxic": []}
    
    # Load activations for each prompt, sorted by toxicity
    for prompt_id, is_toxic in tqdm(labels.items(), desc="Loading activations"):
        category = "toxic" if is_toxic == 1 else "non-toxic"
        
        # Load activations for each layer for this prompt
        for layer in layers:
            file_pattern = f"{activation_type}_layer_{layer}_prompt{prompt_id}.safetensors"
            file_path = os.path.join(activations_dir, file_pattern)
            
            if not os.path.exists(file_path):
                print(f"Warning: Activation file not found for prompt {prompt_id}, layer {layer}: {file_path}")
                continue
            
            # Load the tensor
            tensors = load_file(file_path)
            
            # There should be only one tensor named "activation"
            if "activation" not in tensors:
                print(f"Warning: File {file_path} does not contain an 'activation' tensor. Skipping.")
                continue
            
            # Add this tensor to the appropriate category for this layer
            all_activations[layer][category].append(tensors["activation"].to(device))
    
    # Check if we have enough activations
    for layer, categories in all_activations.items():
        toxic_count = len(categories["toxic"])
        non_toxic_count = len(categories["non-toxic"])
        print(f"Layer {layer}: {toxic_count} toxic activations, {non_toxic_count} non-toxic activations")
        
        if toxic_count == 0 or non_toxic_count == 0:
            print(f"Warning: Layer {layer} has {toxic_count} toxic and {non_toxic_count} non-toxic activations")
    
    return all_activations, labels


def calculate_mean_activations(activations):
    """
    Calculate mean activations for each category (toxic/non-toxic) for each layer.
    
    Args:
        activations: Dictionary of activations by layer, each containing a list of tensors
        
    Returns:
        mean_activations: Dictionary of mean activations by layer and category
    """
    print("Calculating mean activations...")
    mean_activations = {}
    
    for layer, categories in activations.items():
        mean_activations[layer] = {}
        
        for category, tensors in categories.items():
            if not tensors:
                print(f"Warning: No tensors for layer {layer}, category {category}")
                mean_activations[layer][category] = None
                continue
            
            # Stack tensors and compute mean
            try:
                stacked = torch.stack(tensors)
                mean = torch.mean(stacked, dim=0)
                mean_activations[layer][category] = mean
            except Exception as e:
                print(f"Error stacking tensors for layer {layer}, category {category}: {e}")
                
                # Print tensor shapes for debugging
                shapes = [t.shape for t in tensors]
                print(f"Tensor shapes: {shapes}")
                
                # Skip this category
                mean_activations[layer][category] = None
    
    return mean_activations


def calculate_steering_vectors(mean_activations):
    """
    Calculate steering vectors as the difference between non-toxic and toxic mean activations.
    
    Args:
        mean_activations: Dictionary of mean activations by layer and category
        
    Returns:
        steering_vectors: Dictionary of steering vectors by layer
    """
    print("Calculating steering vectors...")
    steering_vectors = {}
    
    for layer, categories in mean_activations.items():
        if categories["toxic"] is None or categories["non-toxic"] is None:
            print(f"Skipping layer {layer} due to missing activations")
            continue
        
        # Calculate steering vector as non-toxic - toxic
        sv = categories["non-toxic"] - categories["toxic"]
        
        # Normalize the steering vector
        norm = torch.norm(sv)
        if norm > 0:
            sv = sv / norm
        
        steering_vectors[layer] = sv
    
    return steering_vectors


def save_steering_vectors(steering_vectors, output_dir, activation_type="hidden_states"):
    """
    Save steering vectors to the specified directory.
    
    Args:
        steering_vectors: Dictionary of steering vectors by layer
        output_dir: Directory to save steering vectors
        activation_type: Type of activations (hidden_states or residual_stream)
    """
    print(f"Saving {activation_type} steering vectors to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each steering vector
    for layer, sv in steering_vectors.items():
        file_name = f"{activation_type}_layer_{layer}_steering_vector.safetensors"
        file_path = os.path.join(output_dir, file_name)
        
        # Save as safetensors
        save_file({"steering_vector": sv.cpu()}, file_path)
        
        print(f"Saved layer {layer} steering vector to {file_path}")


def analyze_steering_vectors(steering_vectors):
    """
    Analyze properties of steering vectors, like norm, sparsity, etc.
    
    Args:
        steering_vectors: Dictionary of steering vectors by layer
        
    Returns:
        analysis: Dictionary of analysis results
    """
    analysis = {}
    
    for layer, sv in steering_vectors.items():
        # Convert to CPU for analysis
        sv_cpu = sv.cpu()
        
        # Calculate norm
        norm = torch.norm(sv_cpu).item()
        
        # Calculate sparsity (percentage of elements close to zero)
        threshold = 0.01  # Consider values less than 1% of max as effectively zero
        max_val = torch.max(torch.abs(sv_cpu)).item()
        zero_threshold = threshold * max_val
        zeros = torch.sum(torch.abs(sv_cpu) < zero_threshold).item()
        sparsity = zeros / sv_cpu.numel()
        
        # Calculate statistics
        mean = torch.mean(sv_cpu).item()
        std = torch.std(sv_cpu).item()
        min_val = torch.min(sv_cpu).item()
        max_val = torch.max(sv_cpu).item()
        
        # Calculate cosine similarity with one-hot vectors
        # This helps identify important dimensions
        
        # Get dimensions with largest magnitudes
        abs_sv = torch.abs(sv_cpu)
        top_dims = torch.topk(abs_sv, k=min(10, len(abs_sv)), dim=0)
        
        top_indices = top_dims.indices.flatten().tolist()
        top_values = top_dims.values.flatten().tolist()
        
        analysis[layer] = {
            "norm": norm,
            "sparsity": sparsity,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "top_dimensions": list(zip(top_indices, top_values)),
            "shape": list(sv_cpu.shape),
        }
    
    return analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate steering vectors from hidden states activations")
    parser.add_argument("--activations_dir", type=str, required=True,
                        help="Directory containing activation files")
    parser.add_argument("--labels_file", type=str, required=True,
                        help="CSV file with toxicity labels")
    parser.add_argument("--output_dir", type=str, default="./code/steering_vectors_hidden_states",
                        help="Directory to save steering vectors")
    parser.add_argument("--activation_type", type=str, default="hidden_states",
                        choices=["hidden_states", "residual_stream"],
                        help="Type of activations to use")
    parser.add_argument("--analysis_file", type=str, default=None,
                        help="Path to save steering vector analysis JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load activations and labels
    activations, labels = load_activations_and_labels(
        args.activations_dir, args.labels_file, args.activation_type
    )
    
    # 2. Calculate mean activations for toxic and non-toxic categories
    mean_activations = calculate_mean_activations(activations)
    
    # 3. Calculate steering vectors
    steering_vectors = calculate_steering_vectors(mean_activations)
    
    # 4. Save steering vectors
    save_steering_vectors(steering_vectors, args.output_dir, args.activation_type)
    
    # 5. Analyze steering vectors
    if args.analysis_file:
        import json
        
        analysis = analyze_steering_vectors(steering_vectors)
        
        with open(args.analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Saved steering vector analysis to {args.analysis_file}")
    
    print("Done!")


if __name__ == "__main__":
    main()
