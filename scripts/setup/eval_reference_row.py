"""Score the shared SFT reference under a config's attribute and save it as the `reference`
row of that config's results.jsonl, without the `run` stage (which would train). Used for
evaluation-only configs such as multipref_formatting_baselines.yaml.
Usage: PYTHONPATH=src python scripts/setup/eval_reference_row.py configs/<exp>.yaml"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import cap_gpu_memory
from data_loading import load_dataset
from evaluation import evaluate_checkpoint
from methods.reference import reference_dir
from utils import load_experiment_config, result_exists, save_result

cfg = load_experiment_config(sys.argv[1])
cap_gpu_memory(cfg)
out = Path(cfg["output_dir"]) / "results.jsonl"
for name in cfg["datasets"]:
    if result_exists(out, "reference", {}, 0):
        print("reference row exists")
        continue
    data = load_dataset(name, cfg)
    res = evaluate_checkpoint(reference_dir(cfg, name) / "model", data, cfg, name, is_reference=True)
    res["dataset"] = name
    save_result(out, "reference", {}, res, 0)
    print("reference", res)
