# README

This repository contains the code for my master’s thesis **“Exploring the Effects of Safety Fine-Tuning in LLM Behaviour.”**  
The project studies how safety fine-tuning changes the internal representations of large language models, and whether harmful behaviours become more structured and controllable after alignment.

The code implements post-hoc activation-level interventions to analyse and manipulate model behaviour without retraining.

## What’s included

- Steering vectors to identify and manipulate directions in activation space associated with undesired behaviour
- Adaptive (per-sample) steering to selectively apply interventions only when needed
- Attention head editing to test whether harmful behaviour is localised in specific heads
- Evaluation tools for:
  - Undesired Output Rate (UOR)
  - Perplexity (to track language quality)
  - Representational similarity

## Purpose

The goal is to compare base vs. safety fine-tuned (instruct) models and test whether safety fine-tuning merely suppresses triggers or fundamentally reshapes the latent space. The code supports experiments showing that instruct models encode undesired behaviour in more structured, steerable representations.

This repository is intended for research and analysis, not for deployment.
