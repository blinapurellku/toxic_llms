#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, json, os, re, numpy as np

ALPHA_RX = re.compile(r"labels_steering_(?P<side>toxic|nontoxic)_alpha_(?P<alpha>[-+]?[\d\.eE]+)\.npy")

def load_all_alphas(d, side="toxic"):
    out = {}
    for p in glob.glob(os.path.join(d, f"labels_steering_{side}_alpha_*.npy")):
        m = ALPHA_RX.search(os.path.basename(p))
        if m and m.group("side")==side:
            out[float(m.group("alpha"))] = np.load(p, allow_pickle=True).item()
    return out

def build_avg_by_layer(alpha2labels):
    alphas = sorted(alpha2labels)
    layers = sorted({k for d in alpha2labels.values() for k in d}, key=lambda x: int(x.split(".")[-1]))
    def avg_valid(v):
        v = [x for x in v if x != -1]
        return float(np.mean(v)) if v else np.nan
    return {L: {a: avg_valid(alpha2labels[a].get(L, [])) for a in alphas} for L in layers}

def extrema(avg_by_layer, baseline, a_min, a_max, eps=0.0):
    def pick_idx(A,V,t):
        idx = [i for i,v in enumerate(V) if np.isclose(v,t,atol=eps,rtol=0.0)]
        return min(idx, key=lambda i:(abs(float(A[i])), float(A[i])))
    res = {}
    for L, per_a in avg_by_layer.items():
        pairs = sorted([(a,v) for a,v in per_a.items() if np.isfinite(v) and a>a_min and a<a_max], key=lambda t:t[0])
        if not pairs: continue
        A,V = zip(*pairs); V = np.asarray(V,float)
        vmax, vmin = float(np.max(V)), float(np.min(V))
        imax, imin = pick_idx(A,V,vmax), pick_idx(A,V,vmin)
        res[L] = dict(
            inc_diff=max(0.0, float(V[imax])-baseline),
            dec_diff=max(0.0, baseline-float(V[imin])),
            alpha_of_max_increase=float(A[imax]),
            avg_toxicity_at_max_increase=float(V[imax]),
            alpha_of_max_decrease=float(A[imin]),
            avg_toxicity_at_max_decrease=float(V[imin]),
            range_avg=float(V[imax]-V[imin]),
        )
    return res

def summarize_layers(results, topk=3):
    if not results: return {}
    gmax = max(v["avg_toxicity_at_max_increase"] for v in results.values())
    gmin = min(v["avg_toxicity_at_max_decrease"] for v in results.values())
    top = sorted(results.items(), key=lambda kv: kv[1]["range_avg"], reverse=True)[:topk]
    inc = max(results.items(), key=lambda kv: kv[1]["inc_diff"])
    dec = max(results.items(), key=lambda kv: kv[1]["dec_diff"])
    top_by_range = [dict(
        layer=L,
        max_avg=m["avg_toxicity_at_max_increase"],
        alpha_up=m["alpha_of_max_increase"],
        min_avg=m["avg_toxicity_at_max_decrease"],
        alpha_down=m["alpha_of_max_decrease"],
        range_avg=m["range_avg"],
    ) for L,m in top]
    return {
        "highest_increase": {"layer": inc[0], **inc[1]},
        "highest_decrease": {"layer": dec[0], **dec[1]},
        "top_by_range": top_by_range,
    }

def one_model(args, model):
    safe = re.sub(r'[\\/*?:"<>|]', "_", model)
    d = os.path.join(args.output_dir, safe)
    os.makedirs(d, exist_ok=True)

    labels_path = os.path.join(d, args.labels_filename)
    if not os.path.exists(labels_path): raise FileNotFoundError(f"Missing baseline labels: {labels_path}")
    labels = np.load(labels_path)
    valid = [x for x in labels if x != -1]
    baseline = float(np.mean(valid)) if valid else np.nan

    alpha2labels = load_all_alphas(d, args.side)
    if not alpha2labels: raise RuntimeError(f"No steering files in {d} for side='{args.side}'.")

    winners = summarize_layers(
        extrema(build_avg_by_layer(alpha2labels), baseline, args.alpha_min, args.alpha_max),
        topk=3
    )

    top3 = winners.get("top_by_range", [])[:3]
    layers      = [winners["highest_increase"]["layer"], winners["highest_decrease"]["layer"], *[t["layer"] for t in top3]]
    alphas_up   = [winners["highest_increase"]["alpha_of_max_increase"], winners["highest_decrease"]["alpha_of_max_increase"], *[t["alpha_up"] for t in top3]]
    alphas_down = [winners["highest_increase"]["alpha_of_max_decrease"], winners["highest_decrease"]["alpha_of_max_decrease"], *[t["alpha_down"] for t in top3]]
    max_avg_tox = [winners["highest_increase"]["avg_toxicity_at_max_increase"], winners["highest_decrease"]["avg_toxicity_at_max_increase"], *[t["max_avg"] for t in top3]]
    min_avg_tox = [winners["highest_increase"]["avg_toxicity_at_max_decrease"], winners["highest_decrease"]["avg_toxicity_at_max_decrease"], *[t["min_avg"] for t in top3]]

    out = {
        model: {
            "layers": layers,            # [layer_max, layer_min, top3...]
            "alphas_up": alphas_up,      # aligned with 'layers'
            "alphas_down": alphas_down,  # aligned with 'layers'
            "max_avg_tox": max_avg_tox,  # aligned with 'layers'
            "min_avg_tox": min_avg_tox,  # aligned with 'layers'
        }
    }

    summary_path = os.path.join(d, args.summary_filename)
    with open(summary_path, "w") as f: json.dump(out, f, indent=2)
    # print(f"[summary] {model} -> {summary_path}")
    print(out)

def parse_args():
    p = argparse.ArgumentParser("Summarize layers & alphas with highest increase/decrease vs baseline in (-2, 2).")
    p.add_argument("--models", nargs="*", default=[
        "allenai/OLMo-2-0425-1B-SFT","allenai/OLMo-2-0425-1B-DPO","allenai/OLMo-2-0425-1B-Instruct",
        "allenai/OLMo-2-0425-1B","google/gemma-2-2b-it","meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-2-2b","meta-llama/Llama-3.2-3B"
    ])
    p.add_argument("--output_dir", default="/data/erblina/Master_thesis")
    p.add_argument("--side", choices=["toxic","nontoxic"], default="toxic")
    p.add_argument("--labels_filename", default="labels.npy")
    p.add_argument("--alpha_min", type=float, default=-2.5)
    p.add_argument("--alpha_max", type=float, default=2.5)
    p.add_argument("--summary_filename", default="diff_extrema_summary.json")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for i, model in enumerate(["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-0425-1B", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"]): #"google/gemma-2-2b-it",

        #  "google/gemma-2-2b-it","meta-llama/Llama-3.2-3B-Instruct",
        # "google/gemma-2-2b","meta-llama/Llama-3.2-3B",
        # "allenai/OLMo-2-0425-1B-SFT","allenai/OLMo-2-0425-1B-DPO","allenai/OLMo-2-0425-1B-Instruct",
        # "allenai/OLMo-2-0425-1B"
        args.model = model
        one_model(args, model)
