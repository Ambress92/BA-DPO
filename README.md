# BA-DPO: Bias-Adjusted Direct Preference Optimization

Code for the paper *BA-DPO: Bias-Adjusted Direct Preference Optimization for Language Model
Alignment*, by Antonio Ferrara, Alberto Rumi and Francesco Bonchi (Intesa Sanpaolo AI
Research): https://arxiv.org/abs/XXXX.XXXXX

It trains and evaluates every arm in the paper through one pipeline: the same data, the same
training loop and the same evaluation, with only the loss changing from arm to arm.

BA-DPO adds a bias term to the DPO logit: `u + theta[idx] . (g_w - g_l)`. Here `u` is the
DPO implicit reward margin, `g` is the declared attribute of each response (for example
"the longer of the two", "contains markdown", "signed with a woman-coded name"), and
`theta` holds the bias parameters. The arms differ only in how `theta` is indexed (see the
table of arms below). The code calls the method BARP-DPO, and its arms `barp_*`, after the
BARP model of annotator bias (cited in the paper) that it puts inside the DPO loss.

## Names used in the code

| Code | Paper |
|---|---|
| `names_v3` (dataset and configs) | Signed-UltraFeedback |
| `multipref` | MultiPref, length declared |
| `multipref_formatting` | MultiPref, formatting (markdown) declared |
| `reference` | Reference (SFT) |
| `dpo` | DPO (BA-DPO with theta frozen at 0) |
| `barp_pooled` | BA-DPO, pooled: one theta per attribute |
| `barp_annotator_mean` | BA-DPO, theta_bar + eps_k: shared mean plus one deviation per annotator |
| `barp_annotator` | BA-DPO, theta_k only |
| `barp_shuffled` | BA-DPO, shuffled ids |
| `barp_class_mean` | BA-DPO, theta_bar + delta_c + eps_k |
| `barp_pooled_warm`, `barp_annotator_mean_warm` | warm-started arms |
| `barp_pooled_gender_only` | pooled, gender declared only |
| `rdpo_a005`, `rdpo` | R-DPO with alpha = 0.005 and alpha = 0.02 |
| `sampo` | SamPO |
| `group_dro_dpo` | Group-DRO DPO |
| `crowd_prefrl` | Crowd-PrefRL |
| `em_minmax_dpo` | EM-DPO with MinMax-DPO |
| `fair_offset*` | DPO with a fixed offset (appendix on the common offset) |

The bias model is in `src/methods/barp_dpo.py`. R-DPO, SamPO, Group-DRO DPO, Crowd-PrefRL
and EM-DPO each have their own module in `src/methods/`. The docstring of `em_dpo.py`
lists where our EM-DPO implementation departs from the original.

## Setup

We used Python 3.14, torch 2.13, transformers 5.16, peft 0.20 and datasets 5.0 on one
NVIDIA A100 80GB.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`scripts/queue.sh` calls `.venv/bin/python`, so create the environment in the repository root.
`meta-llama/Llama-3.1-8B-Instruct` is a gated model on the Hugging Face Hub: request access
and run `huggingface-cli login` before the 8B experiments. All data comes from the Hub
(`allenai/multipref`, `HuggingFaceH4/ultrafeedback_binarized`); nothing is shipped here.

## How the pipeline works

Each experiment is one config in `configs/`. Run every command from the repository root with
`PYTHONPATH=src`.

1. **Build the data.** `python src/experiment_multipref.py generate --config configs/<exp>.yaml`
   writes `data/<data_dir_name>/`. The build is deterministic given `data_seed`.
2. **Train and evaluate.** `bash scripts/queue.sh configs/<exp>.yaml [method:seed ...]`.
   The first run trains the SFT reference of the config. Each (method, seed) pair is then
   trained and evaluated, and one line per pair is appended to
   `experiments/<exp>/results.jsonl`. Without pairs, every method in the config runs at every seed.
   Pairs that already have a result are skipped, so the command can be rerun at any time.
   A run that crashes is tried up to 3 times in all.
3. **Read the tables.** `python scripts/analysis/grid_tables.py [exp ...]` prints each
   table: every cell is the mean over seeds with a 95% interval, and "bias removed" is
   computed per seed against DPO of the same seed.

Evaluation happens inside every run: 300 held-out prompts with sampling at T = 0.7 and
top-p 0.9, the attribute rate, the judge score (Skywork-Reward-V2-Qwen3-1.7B), KL to the
reference, and held-out accuracy on same-group and cross-group pairs. On Signed-UltraFeedback
it also includes the signature read from probabilities and vote prediction with the arm's own
theta.

`python src/experiment_multipref.py run --config ... --method <m> --seeds <s>` does step 2 for
one pair without the queue. `evaluate` scores checkpoints that exist but have no results line.

## Smoke test

```bash
PYTHONPATH=src python src/experiment_multipref.py all --config configs/smoke_names.yaml
```

This builds a toy Signed-UltraFeedback corpus (60 pairs, 6 annotators) and runs every kind of
arm for 3 steps. It checks that the code runs end to end; its numbers mean nothing.

## Reproducing the paper

Seeds are 42, 123 and 456. The same command in the same environment gives the same data and
the same run order. GPU kernels are not bit-deterministic, so a rerun will not reproduce the
paper's numbers to the last digit. On one A100 a 0.5B run takes 70 to 150 minutes and
peaks at 11 to 19 GB. An 8B LoRA run takes 7 to 9 hours (peak 21.5 GB), a Mistral-7B run
4 to 12 hours (peak 17.7 to 23.6 GB).

**Signed-UltraFeedback, 0.5B (main table and variants in the appendix)**
```bash
PYTHONPATH=src python src/experiment_multipref.py generate --config configs/names_v3.yaml
bash scripts/queue.sh configs/names_v3.yaml
```
The config runs every arm at three seeds, except the gender-declared-only arm, which the paper
reports at seed 42. To match it:
`bash scripts/queue.sh configs/names_v3.yaml $(for s in 42 123 456; do for m in dpo barp_pooled barp_annotator_mean barp_class_mean barp_shuffled group_dro_dpo rdpo_a005 sampo em_minmax_dpo; do printf "%s:%s " $m $s; done; done) barp_pooled_gender_only:42`.

**Signed-UltraFeedback, Llama-3.1-8B and Mistral-7B** (after the data build above)
```bash
bash scripts/queue.sh configs/names_v3_llama8b.yaml \
  $(for s in 42 123 456; do printf "dpo:$s barp_pooled:$s barp_annotator_mean:$s group_dro_dpo:$s "; done)
bash scripts/queue.sh configs/names_v3_mistral7b.yaml
```

**MultiPref, length declared (0.5B)**
```bash
PYTHONPATH=src python src/experiment_multipref.py generate --config configs/multipref.yaml
bash scripts/queue.sh configs/multipref.yaml \
  $(for s in 42 123 456; do printf "dpo:$s barp_pooled:$s barp_annotator_mean:$s rdpo_a005:$s sampo:$s "; done)
bash scripts/queue.sh configs/multipref_heterogeneity_baselines.yaml \
  group_dro_dpo:42 group_dro_dpo:123 group_dro_dpo:456 crowd_prefrl:42
```
The variants in the appendix table (theta_k only, shuffled ids, warm starts, R-DPO with
alpha = 0.02) are the remaining methods of `configs/multipref.yaml`. As the table caption
says, they were run against an earlier SFT reference with the same recipe.

**MultiPref, formatting declared (0.5B)**, after the length runs:
```bash
PYTHONPATH=src python src/experiment_multipref.py generate --config configs/multipref_formatting.yaml
bash scripts/setup/link_multipref_formatting.sh          # reuse the length reference, dpo, R-DPO, SamPO
bash scripts/queue.sh configs/multipref_formatting.yaml
PYTHONPATH=src python src/experiment_multipref.py evaluate --config configs/multipref_formatting_baselines.yaml
```
DPO, R-DPO, SamPO and the SFT reference never read the declared attribute, so their length
checkpoints are the formatting checkpoints: the link script reuses them, and only their
evaluation is redone.

**MultiPref at 8B, length and formatting, and Mistral-7B, length** (after the data builds above)
```bash
bash scripts/queue.sh configs/multipref_llama8b.yaml
bash scripts/setup/link_multipref_formatting.sh multipref_llama8b multipref_formatting_llama8b
bash scripts/queue.sh configs/multipref_formatting_llama8b.yaml
bash scripts/queue.sh configs/multipref_mistral7b.yaml
```

**LoRA calibration at 0.5B (appendix)**, after `names_v3`:
```bash
bash scripts/queue.sh configs/names_v3_lora_lr5e-6.yaml
bash scripts/queue.sh configs/names_v3_lora_lr2e-5.yaml
```

**Fixed offset on the biased DPO policy (appendix on the common offset)**, after the
`names_v3` DPO runs. Seed 42 is shown; seeds 123 and 456 use their own configs.
```bash
PYTHONPATH=src python scripts/setup/link_offset_reference.py configs/names_v3_offset_s42.yaml
PYTHONPATH=src python src/experiment_multipref.py generate --config configs/names_v3_offset_s42.yaml
bash scripts/queue.sh configs/names_v3_offset_s42.yaml fair_offset:42 fair_offset_cal_add:42 \
  fair_offset_cal_mul:42 fair_offset_fit_t30:42 fair_offset_fit_t70:42
bash scripts/queue.sh configs/names_v3_offset_s123.yaml fair_offset:123 fair_offset_fit:123
bash scripts/queue.sh configs/names_v3_offset_s456.yaml fair_offset_fit:456
```
The offsets in these configs and how they were derived are written as comments in
`configs/names_v3_offset_s42.yaml`. `scripts/analysis/offset_readouts.py` reads each checkpoint
against the original reference and gives the numbers of the table.

The other scripts in `scripts/analysis/` compute the remaining appendix readouts;
`scripts/README.md` says what each one does.

## Layout

```
configs/            one YAML per experiment; method parameters inline
src/
  experiment_multipref.py   stages: generate, run, evaluate, plot
  data_loading.py   dataset builders (multipref, names = Signed-UltraFeedback, offset pairs)
  attributes.py     the declared attributes g(y, x): length, formatting, name signature
  evaluation.py     every readout, one code path for every arm
  names_readouts.py signature probabilities and vote prediction with theta
  methods/          one module per method, registered in methods/__init__.py
  utils.py          results file, run ids, seeding
scripts/
  queue.sh          trains (method, seed) pairs of one config
  setup/            one-time steps some configs need (links to reused checkpoints)
  analysis/         reads checkpoints or results; never trains
data/               built datasets (created by generate)
experiments/        checkpoints, generations and results.jsonl (created by runs)
```

## Citation

```bibtex
@article{ferrara2026badpo,
  title   = {BA-DPO: Bias-Adjusted Direct Preference Optimization for Language Model Alignment},
  author  = {Ferrara, Antonio and Rumi, Alberto and Bonchi, Francesco},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}
```

## License

CC BY 4.0, see [LICENSE](LICENSE).
