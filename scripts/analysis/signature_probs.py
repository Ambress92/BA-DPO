"""Re-score finished checkpoints with the signature probability readout.

evaluate_checkpoint computes this for every new run (src/names_readouts.py), so this script is
only for checkpoints trained by an earlier version of the pipeline and for the appendix check that the readout does
not depend on the answer body: it scores two bodies per checkpoint, the reference's own answer
(the primary readout, the same text for every arm) and the arm's own answer with its signature
stripped.

Writes <output_dir>/signature_probs/<method>_seed<seed>.json (per-prompt values) and one summary
line per (method, seed) to <output_dir>/signature_probs.jsonl; skips what exists.
Evaluation only, about 3 minutes per checkpoint.
Usage: PYTHONPATH=src python scripts/analysis/signature_probs.py configs/names.yaml
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import cap_gpu_memory, load_frozen, load_tokenizer
from data_loading import load_dataset
from methods.reference import reference_dir
from names_readouts import cell_probs, name_list, strip_sig, summarise
from utils import load_experiment_config


def main():
    cfg = load_experiment_config(sys.argv[1])
    cfg["gpu_memory_cap_gb"] = 6
    cap_gpu_memory(cfg)
    name = cfg["datasets"][0]
    data = load_dataset(name, cfg)                   # activates this build's name pool
    print(f"scoring {len(name_list())} names per prompt", flush=True)
    types = data["meta"]["eval_prompt_types"]
    sign_idx = [i for i, t in enumerate(types) if t == "sign"]
    prompts = [data["eval_prompts"][i] for i in sign_idx]
    ref_dir = reference_dir(cfg, name)
    ref_gens = json.load(open(ref_dir / "generations.json"))
    ref_bodies = [strip_sig(ref_gens[i]["response"]) for i in sign_idx]
    out_dir = Path(cfg["output_dir"]) / "signature_probs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = Path(cfg["output_dir"]) / "signature_probs.jsonl"
    done = set()
    if summary.exists():
        done = {(json.loads(l)["method"], json.loads(l)["seed"]) for l in open(summary)}

    jobs = [("reference", 0, ref_dir / "model", ref_dir / "generations.json")]
    for method in cfg["methods"]:
        if method == "reference":
            continue
        for seed in cfg["seeds"]:
            d = Path(cfg["output_dir"]) / method / f"seed{seed}"
            if (d / "done.json").exists():
                jobs.append((method, seed, d / "model", d / "generations.json"))
    for method, seed, model_dir, gen_path in jobs:
        if (method, seed) in done:
            continue
        tok = load_tokenizer(str(model_dir))
        model = load_frozen(str(model_dir), cfg)
        own_gens = json.load(open(gen_path))
        own_bodies = [strip_sig(own_gens[i]["response"]) for i in sign_idx]
        res = {"ref_body": cell_probs(model, tok, prompts, ref_bodies, cfg),
               "own_body": cell_probs(model, tok, prompts, own_bodies, cfg)}
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(out_dir / f"{method}_seed{seed}.json", "w"))
        rec = {"method": method, "seed": seed,
               "ref_body": summarise(res["ref_body"]), "own_body": summarise(res["own_body"])}
        with open(summary, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(method, seed, json.dumps(rec["ref_body"]), flush=True)


if __name__ == "__main__":
    main()
