"""Creates the reference folder of an offset experiment as a link
to the biased checkpoint named by the config's source_checkpoint, plus the config.json marker
that tells ReferenceSFT.train the reference exists, so no SFT is trained. Safe to run again.
Usage: PYTHONPATH=src .venv/bin/python scripts/setup/link_offset_reference.py configs/names_v3_offset_s42.yaml"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from methods.reference import reference_dir  # noqa: E402
from utils import load_experiment_config  # noqa: E402

cfg = load_experiment_config(sys.argv[1])
src = Path(cfg["source_checkpoint"]).resolve()
assert (src / "config.json").exists(), f"no checkpoint at {src}"
out = reference_dir(cfg, cfg["datasets"][0])
out.mkdir(parents=True, exist_ok=True)
link = out / "model"
if link.is_symlink() or link.exists():
    print(f"{link} exists -> {os.readlink(link) if link.is_symlink() else 'a directory'}")
else:
    link.symlink_to(src)
    print(f"linked {link} -> {src}")
marker = out / "config.json"
if not marker.exists():
    json.dump({"model": str(src), "note": "biased policy used as the reference of the offset stage; not an SFT"},
              open(marker, "w"), indent=1)
    print(f"wrote {marker}")
