"""Readouts of an offset-stage checkpoint against the ORIGINAL names_v3 reference and judgments.
Never trains.
  (i)   KL per token to the SFT reference on the checkpoint's own generations;
  (ii)  signature probabilities with the SFT reference's answer bodies (the table's readout);
  (iii) held-out accuracy on names_v3's held-out judgments: same-group rows are the quality
        preference, which the offset stage must leave unchanged; cross-group rows the attribute
        preference, expected to fall toward the reference's;
  (iv)  the sampling-time tilt, from the SOURCE checkpoint's per-name log-probabilities: the
        log-odds shift added to every woman-coded (Black-coded) name that reaches each target
        rate, found by bisection, and the rate it gives.
Usage (from the repository root, with data/names_v3 and experiments/shared/names_v3 linked to the
names_v3 build and its reference):
  PYTHONPATH=src .venv/bin/python scripts/analysis/offset_readouts.py \
      --checkpoint experiments/names_v3_offset_s42/fair_offset/seed42/model \
      --source experiments/names_v3/dpo/seed42/model --targets 0.5 0.3 0.7"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attributes import CELL_VECTOR
from common import load_frozen, load_tokenizer
from data_loading import load_dataset
from evaluation import generate, heldout_accuracy, kl_to_reference
from methods.reference import reference_dir
from names_readouts import cell_probs, name_list, signature_probabilities, strip_sig
from utils import load_experiment_config

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", help="offset-stage checkpoint (a model folder)")
ap.add_argument("--source", help="the biased source checkpoint, for the sampling-time tilt")
ap.add_argument("--targets", nargs="*", type=float, default=[0.5])
ap.add_argument("--config", default="configs/names_v3.yaml")
ap.add_argument("--n_sign", type=int, default=None, help="use only the first n sign prompts (tests)")
ap.add_argument("--out", help="write the results as json here")
a = ap.parse_args()

cfg = load_experiment_config(a.config)
data = load_dataset("names", cfg)
types = data["meta"]["eval_prompt_types"]
sign = [i for i, t in enumerate(types) if t == "sign"][: a.n_sign]
ref_dir = reference_dir(cfg, "names")
ref_gens = generate(ref_dir / "model", data["eval_prompts"], cfg, ref_dir / "generations.json")
prompts = [data["eval_prompts"][i] for i in sign]
bodies = [strip_sig(ref_gens[i]["response"]) for i in sign]
out = {"n_sign_prompts": len(sign)}

if a.checkpoint:
    ck = Path(a.checkpoint)
    gens = generate(ck, data["eval_prompts"], cfg, ck.parent / "generations.json")
    out["kl_to_sft"] = kl_to_reference(ck, gens, cfg, "names")
    out["sig_p_sft_bodies"] = signature_probabilities(ck, data, cfg, prompts, bodies)
    out["heldout_names_v3"] = heldout_accuracy(ck, data, cfg, "names")

if a.source:
    tok = load_tokenizer(str(a.source))
    model = load_frozen(str(a.source), cfg)
    try:
        per = cell_probs(model, tok, prompts, bodies, cfg)
    finally:
        del model
        torch.cuda.empty_cache()
    vec = np.array([CELL_VECTOR[c] for _, c in name_list()], dtype=float)
    L = np.array([p["logp"] for p in per])                     # (prompts, names)

    def rate(shift, j):
        w = L + shift * vec[:, j][None, :]
        w = w - w.max(1, keepdims=True)
        p = np.exp(w)
        p /= p.sum(1, keepdims=True)
        return float((p * vec[:, j][None, :]).sum(1).mean())

    tilt = {}
    for j, nm in enumerate(data["meta"]["attribute_names"]):
        res = {"rate": rate(0.0, j)}
        for t in a.targets:
            lo, hi = -20.0, 20.0
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if rate(mid, j) < t else (lo, mid)
            s = (lo + hi) / 2
            res[f"target_{t}"] = {"log_odds_shift": s, "c": cfg["beta"] * s, "rate": rate(s, j)}
        tilt[nm] = res
    out["sampling_time_tilt"] = tilt

print(json.dumps(out, indent=1))
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
