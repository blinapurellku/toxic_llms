# Steering Vector Analysis for LLMs

This directory contains a modular workflow for steering vector analysis on Language Models using the HarmBench dataset, with a focus on toxicity reduction. The workflow supports both residual stream analysis (using hooks) and hidden states analysis (using the transformers library's `output_hidden_states=True`).

## Workflow Overview

The workflow consists of the following stages:

1. **Generate and Classify**: Generate model outputs for potentially harmful prompts, classify them for toxicity, and save activations.
2. **Calculate Steering Vectors**: Compute steering vectors as the normalized difference between toxic and non-toxic activations.
3. **Analyze Steering Impact**: Identify the layer where steering vectors have the maximum effect on model outputs.
4. **Generate Steered Outputs**: Apply the best steering vector during generation and evaluate the effect on toxicity.

## Scripts

### Common Workflow (Using Hooks for Residual Stream and Hidden States)

- `generate_and_classify.py`: Generates responses for HarmBench prompts, classifies them for toxicity, and saves activations using hooks.
- `calculate_steering_vectors.py`: Loads activations and toxicity labels, computes steering vectors.
- `analyze_steering_impact.py`: Applies steering vectors to each layer during inference and identifies the most impactful layer.
- `generate_steered.py`: Applies the best steering vector during generation and compares toxicity rates.

### Hidden States Specific Workflow (Using output_hidden_states=True)

- `generate_and_classify_hidden_states.py`: Generates responses and saves hidden states using transformers' `output_hidden_states=True`.
- `calculate_steering_vectors_hidden_states.py`: Computes steering vectors from hidden states activations.
- `analyze_steering_impact_hidden_states.py`: Applies steering vectors to hidden states and identifies the most impactful layer.
- `generate_steered_hidden_states.py`: Applies the best hidden states steering vector during generation.

## Usage Instructions

### 1. Generate and Classify Outputs

#### Using Hooks (Residual Stream and Hidden States)

```bash
python generate_and_classify.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --num_prompts 50 \
    --output_dir "./outputs/llama2-7b-activations" \
    --batch_size 4
```

#### Using Hidden States (No Hooks)

```bash
python generate_and_classify_hidden_states.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --num_prompts 50 \
    --output_dir "./outputs/llama2-7b-hidden-states" \
    --batch_size 4
```

### 2. Calculate Steering Vectors

#### For Residual Stream

```bash
python calculate_steering_vectors.py \
    --activations_dir "./outputs/llama2-7b-activations" \
    --labels_file "./outputs/llama2-7b-activations/toxicity_labels.csv" \
    --output_dir "./outputs/llama2-7b-steering-vectors" \
    --activation_type "residual_stream" \
    --analysis_file "./outputs/llama2-7b-steering-vectors/vector_analysis.json"
```

#### For Hidden States

```bash
python calculate_steering_vectors_hidden_states.py \
    --activations_dir "./outputs/llama2-7b-hidden-states" \
    --labels_file "./outputs/llama2-7b-hidden-states/toxicity_labels.csv" \
    --output_dir "./outputs/llama2-7b-hidden-states-steering-vectors" \
    --activation_type "hidden_states" \
    --analysis_file "./outputs/llama2-7b-hidden-states-steering-vectors/vector_analysis.json"
```

### 3. Analyze Steering Impact

#### For Residual Stream

```bash
python analyze_steering_impact.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --steering_dir "./outputs/llama2-7b-steering-vectors" \
    --output_dir "./outputs/llama2-7b-analysis" \
    --num_test_prompts 10
```

#### For Hidden States

```bash
python analyze_steering_impact_hidden_states.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --steering_dir "./outputs/llama2-7b-hidden-states-steering-vectors" \
    --output_dir "./outputs/llama2-7b-hidden-states-analysis" \
    --num_test_prompts 10
```

### 4. Generate Steered Outputs

#### Using Residual Stream Steering

```bash
python generate_steered.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --steering_dir "./outputs/llama2-7b-steering-vectors" \
    --analysis_dir "./outputs/llama2-7b-analysis" \
    --output_dir "./outputs/llama2-7b-steered-outputs" \
    --num_prompts 25 \
    --scale_factor 5.0
```

#### Using Hidden States Steering

```bash
python generate_steered_hidden_states.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --steering_dir "./outputs/llama2-7b-hidden-states-steering-vectors" \
    --analysis_dir "./outputs/llama2-7b-hidden-states-analysis" \
    --output_dir "./outputs/llama2-7b-hidden-states-steered-outputs" \
    --num_prompts 25 \
    --scale_factor 5.0
```

## Implementation Details

### Activation Storage

All activations are stored using the [safetensors](https://github.com/huggingface/safetensors) format for efficiency and security.

### Reproducibility

All scripts support deterministic generation via random seed setting for reproducible results.

### Batch Processing

The generation scripts support batch processing to handle large prompt datasets efficiently.

### Memory Management

The workflow is designed to minimize memory usage by processing prompts in batches and using memory-efficient tensor operations.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- safetensors
- pandas
- numpy
- tqdm

## Project Structure

```
code/
├── generate_and_classify.py                # Generate outputs and save residual stream
├── calculate_steering_vectors.py           # Calculate steering vectors from activations
├── analyze_steering_impact.py              # Find most impactful layer for steering
├── generate_steered.py                     # Generate with steering applied
├── generate_and_classify_hidden_states.py  # Generate outputs and save hidden states
├── calculate_steering_vectors_hidden_states.py  # Calculate steering from hidden states
├── analyze_steering_impact_hidden_states.py     # Find most impactful hidden states layer
├── generate_steered_hidden_states.py           # Generate with hidden states steering
└── README.md                               # This documentation
```

## Future Work

- Support for more advanced steering techniques (e.g., ablation, compositionality)
- Integration with more diverse datasets beyond HarmBench
- Support for multi-modal models
- Exploration of steering in different modalities (vision, audio)
- Analysis of steering vector transferability across models
