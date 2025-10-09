#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
import numpy as np

# filename pattern: labels_steering_{side}_alpha_{alpha}.npy
ALPHA_RX = re.compile(
    r"labels_steering_(?P<side>toxic|nontoxic)_alpha_(?P<alpha>[-+]?[\d\.eE]+)\.npy"
)

def load_all_alphas(save_dir, side="toxic"):
    """
    Scan `save_dir` for labels_steering_{side}_alpha_*.npy
    Returns: dict[float] -> dict[layer_name] -> list of labels
    """
    alpha_to_layer_labels = {}
    pattern = os.path.join(save_dir, f"labels_steering_{side}_alpha_*.npy")
    for path in glob.glob(pattern):
        m = ALPHA_RX.search(os.path.basename(path))
        if not m or m.group("side") != side:
            continue
        a = float(m.group("alpha"))
        alpha_to_layer_labels[a] = np.load(path, allow_pickle=True).item()
    return alpha_to_layer_labels

def build_avg_by_layer(alpha_to_layer_labels):
    """
    Returns:
      avg_by_layer: dict[layer] -> dict[alpha] -> avg_toxicity
      alphas: sorted list of α
      layers: sorted list of layer names (by numeric suffix)
    """
    alphas = sorted(alpha_to_layer_labels.keys())
    layers = set()
    for d in alpha_to_layer_labels.values():
        layers.update(d.keys())
    layers = sorted(layers, key=lambda x: int(x.split(".")[-1]))

    avg_by_layer = {layer: {} for layer in layers}
    for a, d in alpha_to_layer_labels.items():
        for layer in layers:
            labels = d.get(layer)
            if labels is None:
                continue
            valid = [x for x in labels if x != -1]
            avg = (sum(valid) / len(labels)) if len(labels) > 0 else np.nan
            avg_by_layer[layer][a] = avg
    return avg_by_layer, alphas, layers

def filter_alpha_window(avg_by_layer, alphas, alpha_min=-2.0, alpha_max=2.0):
    """
    Keep only alphas strictly within (alpha_min, alpha_max).
    """
    kept_alphas = [a for a in alphas if (a > alpha_min) and (a < alpha_max)]
    kept = set(kept_alphas)
    filtered = {layer: {a: v for a, v in per_a.items() if a in kept}
                for layer, per_a in avg_by_layer.items()}
    return filtered, sorted(kept_alphas)

def compute_layer_extrema(filtered_avg_by_layer, baseline):
    """
    For each layer compute:
      - max_avg and its α
      - min_avg and its α
      - inc_diff = max(0, max_avg - baseline)
      - dec_diff = max(0, baseline - min_avg)
    Returns: dict[layer] -> metrics
    """
    results = {}
    for layer, per_a in filtered_avg_by_layer.items():
        if not per_a:
            continue
        pairs = [(a, v) for a, v in per_a.items() if np.isfinite(v)]
        if not pairs:
            continue
        alphas, vals = zip(*sorted(pairs, key=lambda t: t[0]))
        arr = np.array(vals, dtype=float)

        max_idx = int(np.argmax(arr))
        min_idx = int(np.argmin(arr))

        max_avg = float(arr[max_idx])
        min_avg = float(arr[min_idx])

        alpha_of_max_increase = float(alphas[max_idx])
        alpha_of_max_decrease = float(alphas[min_idx])

        inc_diff = max(0.0, max_avg - baseline)
        dec_diff = max(0.0, baseline - min_avg)

        results[layer] = {
            "inc_diff": inc_diff,
            "dec_diff": dec_diff,
            "alpha_of_max_increase": alpha_of_max_increase,
            "avg_toxicity_at_max_increase": max_avg,
            "alpha_of_max_decrease": alpha_of_max_decrease,
            "avg_toxicity_at_max_decrease": min_avg,
            "num_alphas_seen": len(alphas),
        }
    return results

def pick_joint_winner(results):
    """
    Pick the single layer that is strongest in BOTH directions.
    Scoring:
      primary = min(inc_diff, dec_diff)  (maximize balance)
      tie-break 1 = max(inc_diff, dec_diff)
      tie-break 2 = inc_diff + dec_diff
      tie-break 3 = numeric layer id (larger wins)
    """
    if not results:
        return None, None

    def layer_id_num(name):
        # expects names like 'model.layers.14'; fallback  -inf if parse fails
        try:
            return int(name.split(".")[-1])
        except Exception:
            return -10**9

    def key_fn(item):
        layer, m = item
        primary = min(m["inc_diff"], m["dec_diff"])
        secondary = max(m["inc_diff"], m["dec_diff"])
        tertiary = (m["inc_diff"] + m["dec_diff"])
        lid = layer_id_num(layer)
        return (primary, secondary, tertiary, lid)

    winner = max(results.items(), key=key_fn)
    return winner[0], winner[1]

def parse_args():
    p = argparse.ArgumentParser("Find the single layer strongest in BOTH increase and decrease within (-2, 2).")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--output_dir", type=str, default="/data/erblina/Master_thesis")
    p.add_argument("--side", choices=["toxic", "nontoxic"], default="toxic")
    p.add_argument("--labels_filename", default="labels.npy")
    p.add_argument("--alpha_min", type=float, default=-2.0)  # strict >
    p.add_argument("--alpha_max", type=float, default=2.0)   # strict <
    p.add_argument("--summary_filename", default="joint_increase_decrease_winner.json")
    return p.parse_args()

def main(args):
    # args = parse_args()
    safe_model = re.sub(r'[\\/*?:"<>|]', "_", args.model)
    model_dir = os.path.join(args.output_dir, safe_model)
    os.makedirs(model_dir, exist_ok=True)

    # baseline
    labels_path = os.path.join(model_dir, args.labels_filename)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing baseline labels: {labels_path}")
    labels = np.load(labels_path)
    valid = [x for x in labels if x != -1]
    baseline = (sum(valid) / len(labels)) if len(labels) > 0 else np.nan
    print(f"[baseline] mean={baseline:.6f}  ({len(valid)}/{len(labels)} valid)")

    # load α files
    alpha_to_layer_labels = load_all_alphas(model_dir, side=args.side)
    if not alpha_to_layer_labels:
        raise RuntimeError(f"No steering files found under {model_dir} for side='{args.side}'.")

    # avg toxicity per layer per α
    avg_by_layer, alphas, layers = build_avg_by_layer(alpha_to_layer_labels)

    # filter to -2 < α < 2 (strict)
    filtered_avg_by_layer, kept_alphas = filter_alpha_window(
        avg_by_layer, alphas, alpha_min=args.alpha_min, alpha_max=args.alpha_max
    )
    if not kept_alphas:
        raise RuntimeError(f"No α within the strict window ({args.alpha_min}, {args.alpha_max}).")
    print(f"[alpha-window] kept {len(kept_alphas)} α in ({args.alpha_min}, {args.alpha_max})")

    # compute per-layer extrema and diffs
    results = compute_layer_extrema(filtered_avg_by_layer, baseline)

    # pick the single joint winner
    layer, metrics = pick_joint_winner(results)
    if layer is None:
        raise RuntimeError("No results after processing (check inputs).")

    # payload with exactly what you asked to see
    out = {
        "model": args.model,
        "side": args.side,
        "alpha_window": {"min_exclusive": args.alpha_min, "max_exclusive": args.alpha_max},
        "baseline_avg_toxicity": baseline,
        "joint_winner": {
            "layer": layer,
            "inc_diff": metrics["inc_diff"],
            "dec_diff": metrics["dec_diff"],
            "alpha_of_max_increase": metrics["alpha_of_max_increase"],
            "avg_toxicity_at_max_increase": metrics["avg_toxicity_at_max_increase"],
            "alpha_of_max_decrease": metrics["alpha_of_max_decrease"],
            "avg_toxicity_at_max_decrease": metrics["avg_toxicity_at_max_decrease"],
            "num_alphas_seen": metrics["num_alphas_seen"],
            "score_primary_min_incdec": min(metrics["inc_diff"], metrics["dec_diff"]),
            "score_secondary_max_incdec": max(metrics["inc_diff"], metrics["dec_diff"]),
            "score_sum_incdec": metrics["inc_diff"] + metrics["dec_diff"],
        },
    }

    # save JSON summary
    summary_path = os.path.join(model_dir, args.summary_filename)
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[summary] wrote {summary_path}")

    # console preview
    jw = out["joint_winner"]
    print(args.model)
    print("\n=== Joint winner (strongest BOTH increase & decrease) ===")
    print(f"Layer: {jw['layer']}")
    print(f"  baseline_avg_toxicity: {baseline:.6f}")
    print(f"  inc_diff: {jw['inc_diff']:.6f}  at α={jw['alpha_of_max_increase']},  "
          f"avg_toxicity@max_increase={jw['avg_toxicity_at_max_increase']:.6f}")
    print(f"  dec_diff: {jw['dec_diff']:.6f}  at α={jw['alpha_of_max_decrease']},  "
          f"avg_toxicity@max_decrease={jw['avg_toxicity_at_max_decrease']:.6f}")
    print(f"  primary score (min inc/dec): {jw['score_primary_min_incdec']:.6f}")


if __name__ == "__main__":
    for i, model in enumerate(["allenai/OLMo-2-0425-1B-SFT", "allenai/OLMo-2-0425-1B-DPO", "allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-0425-1B", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"]): #"google/gemma-2-2b-it",
    # for i, model in enumerate(["allenai/OLMo-2-0425-1B"]):
        args = parse_args()
        args.model = model

        main(args)
