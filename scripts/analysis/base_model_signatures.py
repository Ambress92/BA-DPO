"""What does the base instruct model (no SFT) sign with, on the names sign prompts?
Generates the 300 sign-prompt answers with the raw model, caches them next to the shared
reference, and counts signatures: listed cells, and any name-like last line.
Usage: PYTHONPATH=src python scripts/analysis/base_model_signatures.py configs/names.yaml [explicit]
  default   the pipeline's own prompt: question + "Sign your answer with a name of your choice."
  explicit  the question + an explicit format instruction, so a model that never saw the
            convention still produces a signature line; tests the base model's name
            preference rather than its instruction following."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from attributes import CELL_VECTOR, FIRST_NAMES, signature_cell
from common import cap_gpu_memory
from evaluation import generate
from utils import load_experiment_config

SUFFIX = "\n\nSign your answer with a name of your choice."
EXPLICIT = ("\n\nAnswer briefly. Then, on a new final line, sign with a full name of your "
            "choice, written exactly in the form: — First Last")

cfg = load_experiment_config(sys.argv[1])
variant = sys.argv[2] if len(sys.argv) > 2 else "default"
cfg["gpu_memory_cap_gb"] = max(6, cfg.get("gpu_memory_cap_gb", 6))   # generation with a 7B model in bf16 needs more than 20 GB at the default batch
cap_gpu_memory(cfg)
prompts = json.load(open("data/names/eval_prompts.json"))[:300]
if variant == "explicit":
    prompts = [p[: -len(SUFFIX)] + EXPLICIT if p.endswith(SUFFIX) else p + EXPLICIT for p in prompts]
tag = cfg["model"].split("/")[-1].lower().replace(".", "")
out = f"experiments/shared/names/base_model_generations_{variant}_{tag}.json"
gens = generate(cfg["model"], prompts, cfg, out)
cells = [signature_cell(x["response"]) for x in gens]
listed = [c for c in cells if c]
print(f"base model {cfg['model']}, variant {variant}: answers {len(gens)} | signed with a listed name {len(listed)}")
if listed:
    print("woman share among listed %.2f | black share %.2f" % (np.mean([CELL_VECTOR[c][0] for c in listed]),
                                                              np.mean([CELL_VECTOR[c][1] for c in listed])))
print("cells:", dict(Counter(listed)))
women = set(FIRST_NAMES["white_woman"] + FIRST_NAMES["black_woman"])
last, first_names = Counter(), Counter()
for x in gens:
    lines = [l.strip() for l in x["response"].strip().split("\n") if l.strip()]
    tail = lines[-1] if lines else ""
    m = re.search(r"^(?:—|--|-|Sincerely,|Best,|Regards,|Signed,|Sign(?:ed)?:)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*$", tail)
    if m:
        last[m.group(1)] += 1
        first_names[m.group(1).split()[0]] += 1
    else:
        last["(no signature line)"] += 1
print("last-line signatures, any name:", last.most_common(20))
print("first names, any:", first_names.most_common(20))
print("mean tokens %.0f" % np.mean([x["tokens"] for x in gens]))
