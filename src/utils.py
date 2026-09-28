import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_id(method, params, seed):
    key = json.dumps({"method": method, "params": params, "seed": seed}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def result_exists(path, method, params, seed):
    path = Path(path)
    if not path.exists():
        return False
    target_id = make_run_id(method, params, seed)
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("run_id") == target_id:
                return True
    return False


def save_result(path, method, params, results, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": make_run_id(method, params, seed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_hash(),
        "method": method,
        "seed": seed,
        "params": params,
        "results": results,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_results(path):
    return pd.read_json(path, lines=True)


def save_training_log(path, step_metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log = json.load(open(path)) if path.exists() else []
    log.append(step_metrics)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def load_training_log(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.load(open(path)))


def delete_results(path, method=None, seed=None):
    """Delete specific results. Filters by method and/or seed. Returns count deleted."""
    path = Path(path)
    if not path.exists():
        return 0
    kept, deleted = [], 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            drop = True
            if method is not None and entry.get("method") != method:
                drop = False
            if seed is not None and entry.get("seed") != seed:
                drop = False
            if drop:
                deleted += 1
            else:
                kept.append(line)
    with open(path, "w") as f:
        f.writelines(kept)
    return deleted


def load_experiment_config(path):
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)
    config_dir = path.parent
    resolved = {}
    for name, params in config.get("methods", {}).items():
        if params is None:
            mp = config_dir / "method" / f"{name}.yaml"
            params = (yaml.safe_load(open(mp)) or {}) if mp.exists() else {}
        resolved[name] = params
    config["methods"] = resolved
    return config


def make_latex_table(df, value_col, row_col="method", col_col="dataset",
                     seed_col="seed", fmt=".3f", caption="", label=""):
    grouped = df.groupby([row_col, col_col])[value_col]
    mean = grouped.mean().unstack()
    std = grouped.std().unstack()
    cols, rows = mean.columns.tolist(), mean.index.tolist()
    lines = [r"\begin{table}[t]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{'l' + 'c' * len(cols)}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Method"] + [str(c) for c in cols]) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = [str(row)] + [f"${mean.loc[row, c]:{fmt}} \\pm {std.loc[row, c]:{fmt}}$" for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
