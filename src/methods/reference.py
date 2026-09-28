"""Reference policy: SFT on the majority-chosen response of each comparison.
Trained once per dataset and shared by every method and seed (experiments/shared/<dataset>/reference).
run() is seed-independent: it trains if the checkpoint is missing, otherwise just evaluates."""
import gc
import json
import random
import time

import torch

from common import (SHARED_DIR, collate, cosine_lr, device, encode_pair, load_policy,
                    load_tokenizer, sequence_logps)
from data_loading import majority_chosen
from methods.base import BaseMethod


def reference_dir(cfg, dataset_name):
    # reference_name separates references of different models trained on the same data: by
    # default the folder is named after the data, so an 8B run would find the 0.5B reference
    return SHARED_DIR / cfg.get("reference_name", cfg.get("data_dir_name", dataset_name)) / "reference"


class ReferenceSFT(BaseMethod):
    def __init__(self, params, output_dir=None, cfg=None):
        super().__init__(params, output_dir)
        self.cfg = cfg

    def train(self, data, dataset_name):
        cfg = self.cfg
        out = reference_dir(cfg, dataset_name)
        if (out / "config.json").exists():
            print(f"reference exists at {out}")
            return out
        tok = load_tokenizer(cfg["model"])
        model = load_policy(cfg["model"], cfg, trainable=True)
        rows = majority_chosen(data["train"])
        if cfg.get("sft_resign_random"):
            # names corpora, fair-start reference: the majority
            # vote still selects the answer, but its signature is replaced by a random cell
            # and name, so the reference signs every answer at 50/50 on both attributes.
            from attributes import CELL_VECTOR, FIRST_NAMES, SURNAMES, _SIGNATURE, sign
            rr = random.Random(cfg.get("data_seed", 0) + 1)
            cells = list(CELL_VECTOR)
            for r in rows:
                body = _SIGNATURE.sub("", r["chosen"]).rstrip()
                cell = rr.choice(cells)
                r["chosen"] = sign(body, rr.choice(FIRST_NAMES[cell]), rr.choice(SURNAMES))
            print(f"  sft_resign_random: {len(rows)} answers re-signed with a random cell", flush=True)
        random.Random(0).shuffle(rows)
        seqs = [encode_pair(tok, r["prompt"], r["chosen"], cfg) for r in rows]
        bs, accum = cfg["batch_size"], cfg["grad_accum"]
        steps_per_epoch = len(seqs) // (bs * accum)
        total = steps_per_epoch * cfg["sft_epochs"]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["sft_lr"], weight_decay=0.0)
        model.train()
        log, t0, i = [], time.time(), 0
        for step in range(total):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, cfg["warmup_frac"], cfg["sft_lr"])
            tot_loss = 0.0
            for _ in range(accum):
                batch = seqs[i: i + bs]
                i = (i + bs) % len(seqs)
                ids, att, rm = collate(batch, tok.pad_token_id)
                ids, att, rm = ids.to(device()), att.to(device()), rm.to(device())
                logp, n = sequence_logps(model, ids, att, rm)
                loss = -(logp.sum() / n.sum()) / accum          # token-level NLL
                loss.backward()
                tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            log.append({"step": step, "loss": tot_loss})
            if step % 10 == 0:
                print(f"  sft step {step}/{total} loss {tot_loss:.4f} {(time.time()-t0)/(step+1):.1f}s/step", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        model.to(torch.bfloat16).save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
        json.dump({"model": cfg["model"], "sft_lr": cfg["sft_lr"], "sft_epochs": cfg["sft_epochs"],
                   "n_examples": len(seqs), "steps": total}, open(out / "config.json", "w"))
        json.dump(log, open(out / "training_log.json", "w"))
        del model, opt
        gc.collect()                     # the unmerged LoRA wrapper can sit in a reference cycle
        torch.cuda.empty_cache()
        return out

    def run(self, data, seed):
        raise RuntimeError("ReferenceSFT.train() is called by the experiment script directly")
