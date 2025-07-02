#!/usr/bin/env python3
"""
Calculate steering vectors from activations based on binary toxicity labels.
This script loads activations saved by generate_and_classify.py and calculates
steering vectors as the difference between mean activations of toxic and non-toxic examples.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_classified_responses(csv_path):
    """
    Load the classified responses from CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, sep=";")
    
    # Check if the required column exists
    if "toxicity_label" not in df.columns:
        raise ValueError("CSV file does not contain 'toxicity_label' column")
    
    return df


def load_activations(activation_dir, activation_type):
    """
    Load activations from the specified directory and type.
    Returns a dictionary mapping layer numbers to activation tensors.
    """
    print(f"Loading {activation_type} activations from {activation_dir}...")
    
    # Determine the subdirectory based on activation type
    subdir = os.path.join(activation_dir, activation_type)
    
    if not os.path.exists(subdir):
        raise FileNotFoundError(f"Activation directory not found: {subdir}")
    
    # Get list of all safetensors files
    activation_files = [f for f in os.listdir(subdir) if f.endswith(".safetensors")]
    
    if not activation_files:
        raise ValueError(f"No activation files found in {subdir}")
    
    # Load activations by layer
    layer_activations = {}
    
    for file_name in tqdm(activation_files, desc=f"Loading {activation_type} files"):
        file_path = os.path.join(subdir, file_name)
        
        # Extract layer number from filename (assuming format like "layer_12_hidden.safetensors")
        layer_str = file_name.split("_")[1]
        layer_num = int(layer_str)
        
        # Load the tensors
        tensors = load_file(file_path)
        
        # Store in layer_activations
        layer_activations[layer_num] = tensors
    
    print(f"Loaded activations for {len(layer_activations)} layers")
    return layer_activations


def extract_labeled_activations(layer_activations, labels):
    """
    Split activations by label (toxic vs non-toxic).
    """
    toxic_samples = {}
    non_toxic_samples = {}
    
    print("Splitting activations by toxicity label...")
    
    # For each layer
    for layer_num, tensors in layer_activations.items():
        toxic_samples[layer_num] = []
        non_toxic_samples[layer_num] = []
        
        # For each sample in this layer
        for sample_name, tensor in tensors.items():
            # Extract sample index (assuming format like "sample_42")
            sample_idx = int(sample_name.split("_")[1])
            
            # Skip if sample index is out of bounds
            if sample_idx >= len(labels):
                print(f"Warning: Sample index {sample_idx} exceeds label count {len(labels)}. Skipping.")
                continue
            
            # Get the label for this sample
            label = labels[sample_idx]
            
            # Skip undefined labels (-1)
            if label == -1:
                continue
                
            # Append to appropriate list
            if label == 1:  # Toxic
                toxic_samples[layer_num].append(tensor)
            else:  # Non-toxic
                non_toxic_samples[layer_num].append(tensor)
    
    return toxic_samples, non_toxic_samples


def calculate_mean_vectors(activation_dict):
    """
    Calculate mean activation vectors for each layer.
    """
    mean_vectors = {}
    
    for layer_num, tensors in activation_dict.items():
        if not tensors:  # Skip empty layers
            print(f"Warning: No samples for layer {layer_num}, skipping")
            continue
        
        # Stack tensors to compute mean
        stacked = torch.stack(tensors)
        mean = torch.mean(stacked, dim=0)
        
        mean_vectors[layer_num] = mean
    
    return mean_vectors


def calculate_steering_vectors(toxic_means, non_toxic_means):
    """
    Calculate steering vectors as difference between toxic and non-toxic means.
    """
    steering_vectors = {}
    
    # For each layer present in both dictionaries
    common_layers = set(toxic_means.keys()).intersection(set(non_toxic_means.keys()))
    
    for layer_num in common_layers:
        # Calculate difference vector (toxic - non_toxic)
        steering_vector = toxic_means[layer_num] - non_toxic_means[layer_num]
        
        # Normalize the steering vector
        norm = torch.norm(steering_vector)
        if norm > 0:
            steering_vector = steering_vector / norm
        
        steering_vectors[layer_num] = steering_vector
    
    return steering_vectors


def save_steering_vectors(steering_vectors, output_dir, activation_type):
    """
    Save steering vectors to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each layer's steering vector separately
    for layer_num, vector in steering_vectors.items():
        file_path = os.path.join(output_dir, f"{activation_type}_layer_{layer_num}_steering_vector.safetensors")
        
        # Create a dictionary with a single tensor
        tensor_dict = {f"steering_vector": vector}
        
        # Save to safetensors
        save_file(tensor_dict, file_path)
    
    print(f"Saved {len(steering_vectors)} steering vectors to {output_dir}")
    
    # Create metadata file
    metadata = {
        "activation_type": activation_type,
        "description": "Steering vectors calculated as normalized difference between mean toxic and non-toxic activations",
        "layers": list(steering_vectors.keys()),
        "creation_date": pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(output_dir, f"{activation_type}_steering_vectors_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate steering vectors from activations")
    parser.add_argument("--responses_csv", type=str, required=True, 
                        help="Path to CSV file with classified responses")
    parser.add_argument("--activations_dir", type=str, required=True,
                        help="Directory containing saved activations")
    parser.add_argument("--activation_type", type=str, default="residual_stream",
                        choices=["hidden_states", "residual_stream"],
                        help="Type of activations to use")
    parser.add_argument("--output_dir", type=str, default="./code/steering_vectors",
                        help="Directory to save steering vectors")
    parser.add_argument("--sample_dim", type=int, default=0,
                        help="Dimension along which to average samples")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Load classified responses with toxicity labels
    df = load_classified_responses(args.responses_csv)
    labels = df["toxicity_label"].tolist()
    
    print(f"Loaded {len(labels)} classified responses")
    print(f"Toxic responses: {sum(1 for l in labels if l == 1)}")
    print(f"Non-toxic responses: {sum(1 for l in labels if l == 0)}")
    print(f"Undefined: {sum(1 for l in labels if l == -1)}")
    
    # 2. Load activations
    layer_activations = load_activations(args.activations_dir, args.activation_type)
    
    # 3. Split activations by label
    toxic_samples, non_toxic_samples = extract_labeled_activations(layer_activations, labels)
    
    print(f"Split activations into {sum(len(v) for v in toxic_samples.values())} toxic and "
          f"{sum(len(v) for v in non_toxic_samples.values())} non-toxic samples")
    
    # 4. Calculate mean vectors for each group
    toxic_means = calculate_mean_vectors(toxic_samples)
    non_toxic_means = calculate_mean_vectors(non_toxic_samples)
    
    print(f"Calculated mean vectors for {len(toxic_means)} layers")
    
    # 5. Calculate steering vectors
    steering_vectors = calculate_steering_vectors(toxic_means, non_toxic_means)
    
    print(f"Calculated {len(steering_vectors)} steering vectors")
    
    # 6. Save steering vectors
    save_steering_vectors(steering_vectors, args.output_dir, args.activation_type)


if __name__ == "__main__":
    main()
