# scripts

Run everything from the repository root.

| File | What it does |
|---|---|
| `queue.sh` | The only script that trains. `bash scripts/queue.sh configs/<exp>.yaml [method:seed ...]` runs the given (method, seed) pairs of one config one at a time; without pairs, every method at every seed. Before each run it waits for the config's `gpu_memory_cap_gb` plus 1 GB of free GPU memory. It skips pairs that already have a line in `results.jsonl` or are being run by another queue (locks in `logs/locks/`), and tries a crashed run up to 3 times. `DRY_RUN=1` prints the order and what is already done. Logs: `logs/<exp>.log` and `logs/<exp>_queue.log`. |
| `setup/link_multipref_formatting.sh` | Links the dpo, R-DPO and SamPO checkpoints of a MultiPref length experiment into its formatting experiment, which scores them under formatting without retraining (none of them reads the declared attribute). Arguments: length and formatting experiment names (default: the 0.5B pair). |
| `setup/link_offset_reference.py` | Links the biased DPO checkpoint named in an offset config (`source_checkpoint`) as that experiment's reference, so no SFT is trained. |
| `setup/eval_reference_row.py` | Writes the reference row of an evaluation-only config (no training stage). |
| `analysis/grid_tables.py` | Prints the result tables from `results.jsonl`: every cell the mean over seeds with a 95% interval, bias removed per seed against DPO of the same seed. |
| `analysis/offset_readouts.py` | Reads an offset-stage checkpoint against the original names_v3 reference and judgments (KL to the SFT, rates with the SFT answer bodies, same/cross held-out accuracy) and computes the sampling-time tilt from the source checkpoint. |
| `analysis/signature_probs.py` | Signature probability readout with two answer bodies per checkpoint: the reference's (the readout every run computes) and the arm's own with its signature stripped (the appendix check). |
| `analysis/heldout_with_theta.py` | Vote prediction with the arm's own theta, for checkpoints trained by an earlier version of the pipeline that did not compute it. |
| `analysis/kl_backfill.py` | KL to the reference for checkpoints trained by an earlier version of the pipeline that did not compute it; writes `kl.jsonl`, which `grid_tables.py` reads. |
| `analysis/race_drift_permutation.py` | CPU. Tests whether the sampled race drift of DPO is larger than a random relabelling of the name pool gives. |
| `analysis/base_model_signatures.py` | What the base instruct model signs with, before any fine-tuning. |
