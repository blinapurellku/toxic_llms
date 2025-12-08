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
    EXCLUDED_LAYERS_INDICES = {0, 1}

    def is_layer_excluded(layer_name):
        try:
            # Assuming layer name is like 'block.0' or 'model.layers.4'. We extract the last number.
            layer_idx = int(layer_name.split(".")[-1])
            return layer_idx in EXCLUDED_LAYERS_INDICES
        except ValueError:
            return False # Keep layer if parsing fails
        
    alphas = sorted(alpha2labels)
    # layers = sorted({k for d in alpha2labels.values() for k in d}, key=lambda x: int(x.split(".")[-1]))
    all_layers = {k for d in alpha2labels.values() for k in d}

    # Filter out excluded layers and sort
    layers = sorted(
        [L for L in all_layers if not is_layer_excluded(L)],
        key=lambda x: int(x.split(".")[-1])
    )
    def avg_valid(v):
        v_ = np.array([x for x in v if x != -1])
        if len(v) > 0:
        # v_.sum() is the Count of Toxic Prompts (assuming toxic=1)
        # len(v) is the Total Count of All Prompts (as you requested)
            return float(v_.sum() / len(v))
        else:
            return np.nan
        # return float(avg) if v else np.nan
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
    # gmax = max(v["avg_toxicity_at_max_increase"] for v in results.values())
    # gmin = min(v["avg_toxicity_at_max_decrease"] for v in results.values())
    top = sorted(results.items(), key=lambda kv: kv[1]["range_avg"], reverse=True)[:topk]
    # inc = max(results.items(), key=lambda kv: kv[1]["inc_diff"])
    # dec = max(results.items(), key=lambda kv: kv[1]["dec_diff"])
    top_by_range = [dict(
        layer=L,
        max_avg=m["avg_toxicity_at_max_increase"],
        alpha_up=m["alpha_of_max_increase"],
        min_avg=m["avg_toxicity_at_max_decrease"],
        alpha_down=m["alpha_of_max_decrease"],
        range_avg=m["range_avg"],
    ) for L,m in top]
    return {
        # "highest_increase": {"layer": inc[0], **inc[1]},
        # "highest_decrease": {"layer": dec[0], **dec[1]},
        "top_by_range": top_by_range,
    }

def one_model(args, model):
    safe_model_name = re.sub(r'[\\/*?:"<>|]', "_", args.model)
    # safe_base_name = re.sub(r'[\\/*?:"<>|]', "_", "google/gemma-2-2b")

    theta = args.theta
    side = 'toxic' # or 'nontoxic' 'toxic'
    save_path = os.path.join(args.output_dir, safe_model_name)

    labels_before = np.load(f"{args.output_dir}/{safe_model_name}/labels.npy")
    valid_lab = [r for r in labels_before if r != -1]
    avg_label = sum(valid_lab) / len(labels_before)
    # print(f"Mean toxicity label: {avg_label:.3f}, {sum(valid_lab)}/{len(labels_before)} , valid responses: {len(valid_lab)}")
    # 2) Build a lookup of ALL named modules in the model
    res = {}
    # with open(os.path.join(save_path, "steered_perplexities.json")) as f:
    #     perplexities = json.load(f)
    # # print(perplexities.keys())
    # with open(os.path.join(save_path, "base_perplexity.json")) as f:
    #     base_perplexity = json.load(f)["base_perplexity"]
    # all_p = {}
    
    
    models_get ={
    "allenai/OLMo-2-0425-1B-Instruct":25, "allenai/OLMo-2-0425-1B": 25,"Qwen/Qwen2.5-3B-Instruct":57,"Qwen/Qwen2.5-3B":57,
            "google/gemma-2-2b-it":20,  "google/gemma-2-2b":20, "meta-llama/Llama-3.2-3B-Instruct":67 ,"meta-llama/Llama-3.2-3B":67

    }
    ablate = args.ablate
    fil = args.fil # 'pca' or 'mean_head' or 'diff', 'cosine_diff, cosine

    values = [
    -3.0, -2.8, -2.6,
    -2.5, -2.4, -2.2, -1.8, -1.6, -1.5,
    -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.8, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0, 
    ]
    mode = 'editing'
    res[mode] = {}
    for a in range(5, 16, 1):
    
        
        res[mode][a] = []
        for l in values:
          # range(0, num_heads+1, 2)  # Ablate from 0 to all heads
            top_n = a
            head_id = f'{mode}_top_k_{top_n}_{fil}_lambda_{l}'

            labels_after = np.load(f"{args.output_dir}/editing/{safe_model_name}/all_layers_ablation_head_{head_id}_mean.npy", allow_pickle=True).item()['labels']
            
            valid_lab_editing = [r for r in labels_after if r != -1]
            avg_l = sum(valid_lab_editing) / len(labels_after)

            # res['editing'].append({'top_n': a, 'avg_toxicity': avg_l, 'lambda': l})
            res[mode][a].append({'lambda': l, 'avg_toxicity': avg_l})

    
    results_extrema = {}  # store extrema info per a

    for a, entries in res[mode].items():
        if not entries:  # just in case
            continue

        # highest avg_toxicity
        max_entry = max(entries, key=lambda d: d['avg_toxicity'])
        # lowest avg_toxicity
        min_entry = min(entries, key=lambda d: d['avg_toxicity'])

        results_extrema[a] = {
            'max_avg_toxicity': max_entry['avg_toxicity'],
            'max_lambda': max_entry['lambda'],
            'min_avg_toxicity': min_entry['avg_toxicity'],
            'min_lambda': min_entry['lambda'],
        }

    # Example: print nicely
    for a in sorted(results_extrema):
        info = results_extrema[a]
        # print(
        #     f"a={a}: "
        #     f"max avg={info['max_avg_toxicity']:.4f} at λ={info['max_lambda']}, "
        #     f"min avg={info['min_avg_toxicity']:.4f} at λ={info['min_lambda']}"
        # )

    best_a = None
    best_score = -float('inf')

    for a, info in results_extrema.items():
        score = info['max_avg_toxicity'] - info['min_avg_toxicity']
        if score > best_score:
            best_score = score
            best_a = a

    # Retrieve the lambdas
    overall_best = {}

    overall_best[model] = {
        
        'a': best_a,
        'min_lambda': results_extrema[best_a]['min_lambda'],
        'min_avg_toxicity': results_extrema[best_a]['min_avg_toxicity'],
        'max_lambda': results_extrema[best_a]['max_lambda'],
        'max_avg_toxicity': results_extrema[best_a]['max_avg_toxicity'],
    }
    # print(overall_best)
    return overall_best


    


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
    p.add_argument("--alpha_min", type=float, default=-3.5)
    p.add_argument("--alpha_max", type=float, default=3.5)
    p.add_argument("--summary_filename", default="diff_extrema_summary.json")
    p.add_argument("--ablate", action="store_true", help="Whether to perform ablation analysis")
    p.add_argument("--theta", type=float, default=0.3, help="Theta value for ablation analysis")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = {}
    for i, model in enumerate(["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-0425-1B", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b", "meta-llama/Llama-3.2-3B"]): #"google/gemma-2-2b-it",

        #  "google/gemma-2-2b-it","meta-llama/Llama-3.2-3B-Instruct",
        # "google/gemma-2-2b","meta-llama/Llama-3.2-3B",
        # "allenai/OLMo-2-0425-1B-SFT","allenai/OLMo-2-0425-1B-DPO","allenai/OLMo-2-0425-1B-Instruct",
        # "allenai/OLMo-2-0425-1B"
        args.ablate = False
        # args.theta = 0.5
        args.fil = 'distance' #'mean_sv'
        args.model = model
        o = one_model(args, model)
        res.update(o)
    print(res)