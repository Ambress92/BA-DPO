"""KL to the reference for checkpoints whose results.jsonl row lacks it.

evaluate_checkpoint computes KL for every run, so this script is only for checkpoints trained by
an earlier version of the pipeline, which did not compute it. It calls the pipeline's kl_to_reference on
the generations the run already saved, so the value is the one the pipeline would have written.

Writes one line per (method, seed) to <output_dir>/kl.jsonl; skips what exists there and rows
that already have KL. Evaluation only, about 2 minutes per 0.5B checkpoint.
Usage: .venv/bin/python scripts/analysis/kl_backfill.py configs/multipref.yaml
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import cap_gpu_memory
from evaluation import kl_to_reference
from utils import load_experiment_config


def main():
    cfg = load_experiment_config(sys.argv[1])
    cap_gpu_memory(cfg)
    name = cfg["datasets"][0]
    out = Path(cfg["output_dir"])
    have = {(e["method"], e["seed"]) for e in map(json.loads, open(out / "results.jsonl"))
            if e["results"].get("kl_per_token") is not None}
    summary = out / "kl.jsonl"
    if summary.exists():
        have |= {(e["method"], e["seed"]) for e in map(json.loads, open(summary))}
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            d = out / method / f"seed{seed}"
            if method == "reference" or (method, seed) in have or not (d / "done.json").exists():
                continue
            res = kl_to_reference(d / "model", json.load(open(d / "generations.json")), cfg, name)
            with open(summary, "a") as f:
                f.write(json.dumps({"method": method, "seed": seed, **res}) + "\n")
            print(method, seed, json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
