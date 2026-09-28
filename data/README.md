# Datasets

Nothing is shipped here. Every dataset is built from public Hugging Face data by the `generate`
stage, deterministically given `data_seed`:

```bash
PYTHONPATH=src python src/experiment_multipref.py generate --config configs/<exp>.yaml
```

| Folder | Built by | Source |
|---|---|---|
| `multipref` | `configs/multipref.yaml` | `allenai/multipref`, split `train` |
| `multipref_formatting` | `configs/multipref_formatting.yaml` | the same, with the formatting attribute |
| `names_v3` (Signed-UltraFeedback) | `configs/names_v3.yaml` | `HuggingFaceH4/ultrafeedback_binarized`, split `train_prefs`, plus simulated annotators |
| `names_v3_selfpairs_s<seed>` | `configs/names_v3_offset_s<seed>.yaml` | generations of the names_v3 DPO checkpoint of that seed |
| `names_smoke` | `configs/smoke_names.yaml` | toy version of names_v3 |

`multipref`: every annotator judgment is one row (4 per comparison, never aggregated). Ties are
dropped. Annotator ids are mapped to 0..m-1, and the mapping is saved in `meta.json`. The held-out
split is by prompt. The attribute column comes from the group function named in the config
(`length` is pairwise: the response at least `length_ratio` times longer than its partner gets g = 1).

`names_v3`: UltraFeedback pairs in which answers carry a signature line whose first name is
gender- and race-coded (`src/attributes.py`). 60 simulated annotators in 3 classes, each with a
known bias vector theta_k = class mean + N(0, theta_sigma^2) per attribute, vote on each pair
through the BARP likelihood. The planted theta_k and the realised class means are saved in
`meta.json`.
