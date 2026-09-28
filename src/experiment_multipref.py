"""BARP-DPO experiment driver. Stages: generate (data), run (train + evaluate), evaluate
(score any checkpoint that lacks a results entry), plot."""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import cap_gpu_memory
from data_loading import BUILDERS, DATA_DIR, load_dataset
from evaluation import evaluate_checkpoint
from methods import get_method
from methods.reference import ReferenceSFT, reference_dir
from utils import load_experiment_config, load_results, result_exists, save_result, set_seed


def generate(config):
    for name in config["datasets"]:
        if (DATA_DIR / config.get("data_dir_name", name) / "meta.json").exists():
            print(f"data/{config.get('data_dir_name', name)} exists")
            continue
        BUILDERS[name](config)


def _ensure_reference(config, data, dataset_name):
    ReferenceSFT({}, cfg=config).train(data, dataset_name)
    out = Path(config["output_dir"]) / "results.jsonl"
    if not result_exists(out, "reference", {}, 0):
        res = evaluate_checkpoint(reference_dir(config, dataset_name) / "model", data, config, dataset_name,
                                  is_reference=True)
        res["dataset"] = dataset_name
        save_result(out, "reference", {}, res, 0)
        print(f"  Done: reference {res}")


def run(config):
    out = Path(config["output_dir"]) / "results.jsonl"
    errors = []
    for dataset_name in config["datasets"]:
        data = load_dataset(dataset_name, config)
        _ensure_reference(config, data, dataset_name)
        for name, params in config["methods"].items():
            if name == "reference":
                continue
            method = get_method(name, params, output_dir=Path(config["output_dir"]) / name, cfg=config)
            for seed in config["seeds"]:
                if result_exists(out, name, params, seed):
                    print(f"  Skipping (already exists): method={name} seed={seed}")
                    continue
                try:
                    set_seed(seed)
                    train = method.run(data, seed, dataset_name=dataset_name)
                    res = method.evaluate(data, seed, dataset_name)
                    res.update(train)
                    res["dataset"] = dataset_name
                    save_result(out, name, params, res, seed)
                    print(f"  Done: method={name} seed={seed} {json.dumps(res)}", flush=True)
                except Exception:
                    msg = f"FAILED: method={name} dataset={dataset_name} seed={seed}"
                    print(f"  {msg}")
                    traceback.print_exc()
                    errors.append(msg)
    if errors:
        print(f"\n{len(errors)} run(s) failed:")
        for e in errors:
            print(f"  - {e}")


def evaluate(config):
    """Score checkpoints on disk that have no results entry (e.g. after a judge change)."""
    out = Path(config["output_dir"]) / "results.jsonl"
    for dataset_name in config["datasets"]:
        data = load_dataset(dataset_name, config)
        for name, params in config["methods"].items():
            if name == "reference":
                continue
            method = get_method(name, params, output_dir=Path(config["output_dir"]) / name, cfg=config)
            for seed in config["seeds"]:
                d = method.output_dir / f"seed{seed}"
                if not (d / "done.json").exists() or result_exists(out, name, params, seed):
                    continue
                res = method.evaluate(data, seed, dataset_name)
                res.update(json.load(open(d / "done.json")))
                res["dataset"] = dataset_name
                save_result(out, name, params, res, seed)
                print(f"  Evaluated: method={name} seed={seed}")


def plot(config):
    path = Path(config["output_dir"]) / "results.jsonl"
    if not path.exists():
        print(f"No results at {path}. Run the experiment first.")
        return
    import pandas as pd
    df = load_results(path)
    r = pd.json_normalize(df["results"])
    r["method"], r["seed"] = df["method"], df["seed"]
    cols = ["attr_rate", "attr_rate_std", "mean_tokens", "reward_raw", "reward_inv",
            "heldout_acc_all", "heldout_acc_same_group", "heldout_acc_cross_group", "theta_mean", "theta_std",
            "kl_per_token"]
    cols = [c for c in cols if c in r]
    r[cols] = r[cols].apply(pd.to_numeric, errors="coerce")   # vector attributes store lists; skip them here
    summary = r.groupby("method")[cols].agg(["mean", "std"]).round(3)
    print(summary.to_string())
    Path("plots").mkdir(exist_ok=True)
    summary.to_csv(Path("plots") / f"{config['experiment_name']}_summary.csv")


def main():
    p = argparse.ArgumentParser(description="BARP-DPO")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["generate", "run", "evaluate", "plot", "all"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/multipref.yaml")
        sp.add_argument("--method", help="Run only this method")
        sp.add_argument("--seeds", nargs="*", type=int)
        sp.add_argument("--override", nargs="*", default=[], help="key=value config overrides (smoke tests)")
    args = p.parse_args()
    config = load_experiment_config(args.config)
    cap_gpu_memory(config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        config[k] = json.loads(v) if v[0] in "0123456789-[{\"tfn" else v
    if args.method:
        config["methods"] = {args.method: config["methods"][args.method]}
    if args.seeds:
        config["seeds"] = args.seeds
    stages = {"generate": generate, "run": run, "evaluate": evaluate, "plot": plot}
    for s in (stages if args.command == "all" else [args.command]):
        stages[s](config)


if __name__ == "__main__":
    main()
