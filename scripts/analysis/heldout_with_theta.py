"""Re-score finished checkpoints with held-out vote prediction under the arm's own theta.

evaluate_checkpoint computes this for every new run (src/names_readouts.py), so this script is
only for checkpoints trained by an earlier version of the pipeline. It scores sign(u + theta_k . (g_c - g_r)) on the
held-out judgments, split by the voter's class, with theta = 0 (the policy alone) and the planted
theta_k (the ceiling) as anchors.

Writes one JSON line per (method, seed) to <output_dir>/heldout_theta.jsonl; skips entries already
there. Evaluation only, about 6 minutes per checkpoint.
Usage: PYTHONPATH=src python scripts/analysis/heldout_with_theta.py configs/names.yaml
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import cap_gpu_memory
from data_loading import load_dataset
from names_readouts import accuracies, per_annotator_theta, policy_margin
from utils import load_experiment_config


def main():
    cfg = load_experiment_config(sys.argv[1])
    cap_gpu_memory(cfg)
    name = cfg["datasets"][0]
    data = load_dataset(name, cfg)
    meta, rows = data["meta"], data["heldout"]
    classes = list(range(len(meta["class_names"])))
    out_path = Path(cfg["output_dir"]) / "heldout_theta.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["method"], json.loads(l)["seed"]) for l in open(out_path)}
    theta_true = np.array(meta["theta_true"], dtype=float)
    for method, params in cfg["methods"].items():
        if method == "reference":
            continue
        for seed in cfg["seeds"]:
            d = Path(cfg["output_dir"]) / method / f"seed{seed}"
            if not (d / "done.json").exists() or (method, seed) in done:
                continue
            u = policy_margin(d / "model", rows, cfg, name)
            rec = {"method": method, "seed": seed,
                   "policy_only": accuracies(u, rows, np.zeros_like(theta_true), classes),
                   "with_own_theta": accuracies(u, rows, per_annotator_theta(d, params, meta), classes),
                   "with_true_theta": accuracies(u, rows, theta_true, classes)}
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(method, seed, json.dumps(rec["with_own_theta"]), flush=True)


if __name__ == "__main__":
    main()
