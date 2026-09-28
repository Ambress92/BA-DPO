"""Permutation test for the DPO race drift on the names corpora (CPU only, no models).

Question: DPO raises the Black-coded signature rate over the reference (+0.17 on the first
build). DPO also piles a third of its signatures on three names (Latoya, Lakisha, Latonya).
Is the drift a property of the Black-coded name list, or would any random half of the pool
show a shift of that size just because DPO concentrates on a few names?

Null: re-label the pool at random. Two versions.
  halves      any 18 (or 44/45) names out of the pool count as "Black-coded";
  stratified  half of the women's names and half of the men's names, so the gender push
              (which is real and shared) cannot leak into the statistic.
Statistic: mean over the 300 sign prompts of 1[first name in the relabelled set], for DPO
(mean over seeds) minus the reference. Raw rates, not length-standardised.

Usage: .venv/bin/python scripts/analysis/race_drift_permutation.py configs/names.yaml [n_perm]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from attributes import FIRST_NAMES, _SIGNATURE     # noqa: E402
from data_loading import load_dataset              # noqa: E402
from methods.reference import reference_dir        # noqa: E402
from utils import load_experiment_config           # noqa: E402


def first_names(gen_path, sign_idx):
    gens = json.load(open(gen_path))
    out = []
    for i in sign_idx:
        m = _SIGNATURE.search(gens[i]["response"])
        out.append(m.group(1) if m else None)
    return out


def main():
    cfg = load_experiment_config(sys.argv[1])
    n_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    name = cfg["datasets"][0]
    data = load_dataset(name, cfg)                     # activates the build's name pool
    types = data["meta"]["eval_prompt_types"]
    sign_idx = [i for i, t in enumerate(types) if t == "sign"]
    pool = {cell: list(ns) for cell, ns in FIRST_NAMES.items()}
    all_names = [n for ns in pool.values() for n in ns]
    women = pool["white_woman"] + pool["black_woman"]
    men = pool["white_man"] + pool["black_man"]
    black = set(pool["black_woman"] + pool["black_man"])
    print(f"build {name}: pool {len(all_names)} names, {len(black)} Black-coded; {len(sign_idx)} sign prompts")

    ref = first_names(reference_dir(cfg, name) / "generations.json", sign_idx)
    dpo = {}
    for seed in cfg["seeds"]:
        p = Path(cfg["output_dir"]) / "dpo" / f"seed{seed}" / "generations.json"
        if p.exists():
            dpo[seed] = first_names(p, sign_idx)
    print(f"dpo seeds: {sorted(dpo)}")

    def rate(names_list, S):
        return float(np.mean([n in S for n in names_list]))

    def stat(S):
        return float(np.mean([rate(d, S) for d in dpo.values()]) - rate(ref, S))

    obs = stat(black)
    print(f"observed: reference {rate(ref, black):.3f}, dpo per seed "
          f"{[round(rate(d, black), 3) for d in dpo.values()]}, drift {obs:+.3f}")
    c = Counter(n for d in dpo.values() for n in d if n)
    top = c.most_common(6)
    tot = sum(c.values())
    print("dpo top names (share of signed):", [(n, round(k / tot, 3)) for n, k in top])
    top3 = {n for n, _ in top[:3]}
    print(f"drift with the top three dpo names removed from the Black-coded set: "
          f"{stat(black - top3):+.3f}")
    print(f"in-pool share: reference {np.mean([n in all_names for n in ref]):.3f}, "
          f"dpo {np.mean([n in all_names for d in dpo.values() for n in d]):.3f}")

    rng = np.random.default_rng(0)
    k = len(black)
    nulls = {"halves": [], "stratified": []}
    for _ in range(n_perm):
        S = set(rng.choice(all_names, size=k, replace=False))
        nulls["halves"].append(stat(S))
        S = set(rng.choice(women, size=len(pool["black_woman"]), replace=False)) | \
            set(rng.choice(men, size=len(pool["black_man"]), replace=False))
        nulls["stratified"].append(stat(S))
    out = {"build": name, "observed": obs, "n_perm": n_perm, "seeds": sorted(dpo)}
    for key, v in nulls.items():
        v = np.array(v)
        p_one = float((v >= obs).mean())
        p_two = float((np.abs(v) >= abs(obs)).mean())
        q = np.quantile(v, [0.025, 0.5, 0.975])
        print(f"null {key:10s}: mean {v.mean():+.3f} sd {v.std():.3f} "
              f"2.5/50/97.5% {q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f}  "
              f"P(null >= obs) {p_one:.3f}  P(|null| >= |obs|) {p_two:.3f}  "
              f"z {(obs - v.mean()) / v.std():.2f}")
        out[key] = {"mean": float(v.mean()), "sd": float(v.std()), "q": q.tolist(),
                    "p_one_sided": p_one, "p_two_sided": p_two}
    Path(cfg["output_dir"], "race_drift_permutation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
