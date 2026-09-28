"""Tables of the main results: one table per experiment, each cell the mean
over seeds with a 95% confidence interval (t distribution, n - 1 degrees of freedom). Bias
removed is computed per seed against DPO of the same seed, (DPO - arm) / (DPO - reference), then
averaged. The reference is one checkpoint, so it has no interval. Reads results.jsonl only.

Usage: .venv/bin/python scripts/analysis/grid_tables.py [experiment ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ["names_v3", "multipref", "multipref_formatting",
           "names_v3_llama8b", "multipref_llama8b", "multipref_formatting_llama8b"]
ARMS = ["reference", "dpo", "barp_pooled", "barp_annotator_mean", "rdpo_a005", "sampo",
        "group_dro_dpo", "em_minmax_dpo"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def gap(r):
    s, c = r.get("heldout_acc_same_group"), r.get("heldout_acc_cross_group")
    return None if s is None or c is None else c - s


def votes(key, cls=None):
    def f(r):
        v = r.get("heldout_own_theta", {})
        return v.get(key) if cls is None else (v.get(key) or [None] * 3)[cls]
    return f


# (column, value from a result row, decimals, whether a bias-removed column follows)
METRICS = {
    "names": [("p(woman)", lambda r: r.get("sig_p_woman"), 3, True),
              ("p(Black)", lambda r: r.get("sig_p_black"), 3, True),
              ("RM (sig. removed)", lambda r: r.get("reward_inv"), 2, False),
              ("KL", lambda r: r.get("kl_per_token"), 4, False),
              ("race votes", votes("attr1_cross"), 3, False),
              ("race votes, class B", votes("attr1_cross_by_class", 1), 3, False),
              ("gender votes", votes("attr0_cross"), 3, False)],
    "length": [("tokens", lambda r: r.get("mean_tokens"), 1, True),
               ("held-out gap", gap, 3, False),
               ("RM", lambda r: r.get("reward_raw"), 2, False),
               ("KL", lambda r: r.get("kl_per_token"), 4, False)],
    "formatting": [("rate (len-std)", lambda r: r.get("attr_rate_std"), 3, True),
                   ("held-out gap", gap, 3, False),
                   ("RM (markup removed)", lambda r: r.get("reward_inv"), 2, False),
                   ("KL", lambda r: r.get("kl_per_token"), 4, False)],
}


def cell(values, decimals):
    v = [x for x in values if x is not None]
    if not v:
        return "-"
    m = float(np.mean(v))
    if len(v) == 1:
        return f"{m:.{decimals}f} (1 seed)"
    half = T95[len(v) - 1] * float(np.std(v, ddof=1)) / np.sqrt(len(v))
    return f"{m:.{decimals}f} ± {half:.{decimals}f}"


def table(exp):
    path = ROOT / "experiments" / exp / "results.jsonl"
    if not path.exists():
        return f"## {exp}\n\nno results yet\n"
    runs = {}
    for line in open(path):
        e = json.loads(line)
        runs.setdefault(e["method"], {})[e["seed"]] = e["results"]
    # rows written by an earlier version of the pipeline keep the two names readouts in side
    # files (scripts/analysis/signature_probs.py and heldout_with_theta.py) and KL in kl.jsonl
    # (scripts/analysis/kl_backfill.py); fill in what is missing
    for fname, key in (("signature_probs.jsonl", "sig_p_woman"), ("heldout_theta.jsonl", "heldout_own_theta"),
                       ("kl.jsonl", "kl_per_token")):
        side = path.parent / fname
        for e in (map(json.loads, open(side)) if side.exists() else []):
            r = runs.get(e["method"], {}).get(e["seed"])
            if r is None or r.get(key) is not None:
                continue
            if key == "kl_per_token":
                r.update(kl_per_token=e["kl_per_token"], kl_per_response=e["kl_per_response"])
            elif key == "sig_p_woman":
                r.update(sig_p_woman=e["ref_body"]["p_woman"], sig_p_black=e["ref_body"]["p_black"])
            else:
                r["heldout_own_theta"] = e["with_own_theta"]
    kind = "names" if "names" in exp else "formatting" if "formatting" in exp else "length"
    ref = next(iter(runs.get("reference", {}).values()), None)
    dpo = runs.get("dpo", {})
    header = ["arm", "seeds"]
    for name, _, _, removed in METRICS[kind]:
        header += [name] + (["removed %"] if removed else [])
    lines = [f"## {exp}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm in ARMS:
        if arm not in runs:
            continue
        seeds = sorted(runs[arm])
        row = [arm, ",".join(str(s) for s in seeds)]
        for _, get, dec, removed in METRICS[kind]:
            row.append(cell([get(runs[arm][s]) for s in seeds], dec))
            if removed:
                rem = []
                if arm not in ("reference", "dpo") and ref is not None:
                    for s in seeds:
                        a, d, r0 = get(runs[arm][s]), get(dpo[s]) if s in dpo else None, get(ref)
                        if None not in (a, d, r0) and d != r0:
                            rem.append(100 * (d - a) / (d - r0))
                row.append(cell(rem, 0))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for exp in sys.argv[1:] or DEFAULT:
        print(table(exp))
