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
    print(f"Mean toxicity label: {avg_label:.3f}, {sum(valid_lab)}/{len(labels_before)} , valid responses: {len(valid_lab)}")
    # 2) Build a lookup of ALL named modules in the model
    res = {}
    # with open(os.path.join(save_path, "steered_perplexities.json")) as f:
    #     perplexities = json.load(f)
    # # print(perplexities.keys())
    # with open(os.path.join(save_path, "base_perplexity.json")) as f:
    #     base_perplexity = json.load(f)["base_perplexity"]
    # all_p = {}
    res['mitigate'] = []
    res['amplify'] = []
    
    models_get ={
    "allenai/OLMo-2-0425-1B-Instruct":25, "allenai/OLMo-2-0425-1B": 25,"Qwen/Qwen2.5-3B-Instruct":57,"Qwen/Qwen2.5-3B":57,
            "google/gemma-2-2b-it":20,  "google/gemma-2-2b":20, "meta-llama/Llama-3.2-3B-Instruct":67 ,"meta-llama/Llama-3.2-3B":67

    }
    ablate = args.ablate
    fil ='cosine' # 'pca' or 'mean_head' or 'diff', 'cosine_diff, cosine
    for a in range(1, models_get[args.model]+1):  # range(0, num_heads+1, 2)  # Ablate from 0 to all heads
        # if a == 29 or a == 46 and args.model == "Qwen/Qwen2.5-3B-Instruct":
        #     continue

        layer_name = "all_layers"
        # head_id = f"mitigate_top_k_{a}_non"  #
        top_n = a
        mode = 'mitigate'
        head_id = f'{mode}_top_k_{top_n}_{fil}'#_non' 
        if ablate:
            addin = 'ablate'
        else:
            addin = f't_theta_{theta}' # f'theta_{0.5}' or 'ablate' # we need the t_ because it is choosing by projecting on the toxic side


        labels_after = np.load(f"{args.output_dir}/mitigate/{safe_model_name}/all_layers_ablation_head_{head_id}_{addin}.npy", allow_pickle=True).item()['labels']
        mode = 'amplify'
        head_id = f'{mode}_top_k_{top_n}_{fil}'
        labels_after_amplify = np.load(f"{args.output_dir}/amplify/{safe_model_name}/all_layers_ablation_head_{head_id}_{addin}.npy", allow_pickle=True).item()['labels']
        valid_lab_mitigate = [r for r in labels_after if r != -1]
        avg_l = sum(valid_lab_mitigate) / len(labels_after)
        res['mitigate'].append({'top_n': a, 'avg_toxicity': avg_l})
        
        valid_lab_amplify = [r for r in labels_after_amplify if r != -1]
        avg_l_amplify = sum(valid_lab_amplify) / len(labels_after_amplify)
        res['amplify'].append({'top_n': a, 'avg_toxicity': avg_l_amplify})

    res_amp = sorted(res['amplify'], key=lambda x: x['top_n'])
    res_mit = sorted(res['mitigate'], key=lambda x: x['top_n'])

    amp_vals = np.array([r['avg_toxicity'] for r in res_amp])
    mit_vals = np.array([r['avg_toxicity'] for r in res_mit])
    top_n_vals = np.array([r['top_n'] for r in res_amp])  # assuming same top_n set

    # Compute differences
    diff = amp_vals - mit_vals

    # ✅ Condition: only consider cases where amplify > mitigate
    mask = diff > 0
    if np.any(mask):
        max_idx = np.argmax(diff * mask)  # this works because masked negatives become 0
        best_top_n = top_n_vals[max_idx]
        best_diff = diff[max_idx]
        print(f"Largest positive difference at top_n={best_top_n}: Δtoxicity={best_diff:.4f}")
    else:
        print("No top_n where amplify > mitigate.")
    
    return model, best_top_n
    

    # print(f"Head ablation top {a} mitigate - Mean toxicity label: {avg_l}")
    # print(f"Head ablation top {a} amplify - Mean toxicity label: {avg_l_amplify}")

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
        args.ablate = True
        args.theta = 0.5
        print(model, args.theta)
        args.model = model
        mo, best_n = one_model(args, model)
        res[mo] = best_n.item()
    print(args.ablate, args.theta)
    print("Final Results:", res)
