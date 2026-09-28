"""BARP-DPO. One training loop, four arms differing only in the bias index:
  dpo             theta frozen at 0
  barp_pooled     one theta shared by every row
  barp_annotator  theta_k per annotator
  barp_shuffled   theta_k per annotator, but each row's annotator id is drawn at random
                  (seeded), so the ids carry no information about the labeller.
Logit = beta * [(log pi - log ref)(y_w) - (log pi - log ref)(y_l)] + theta[idx] . (g_w - g_l).
The attribute may be a vector (data meta `attribute_names`); theta is then a matrix with one
column per attribute and the bias term is a scalar product. Scalar attributes are the d=1
case and produce exactly the previous numbers. Further params:
  n_bias_params: per_class     one theta per annotator class (rows carry `annotator_class`)
  shared_mean + class_level    theta_k = theta_bar + delta_class(k) + eps_k
  declared_attributes: [...]   train on that subset of the attributes only (others masked to 0)
Reference log-probs are cached once per dataset (they never change across seeds or arms).
"""
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (SHARED_DIR, adapters_disabled, collate, cosine_lr, device, encode_pair,
                    load_frozen, load_policy, load_tokenizer, sequence_logps)
from methods.base import BaseMethod
from methods.reference import reference_dir


def _ref_cache_path(cfg, dataset_name, split):
    tag = f"{split}_L{cfg['max_length']}_P{cfg['max_prompt_length']}"
    return reference_dir(cfg, dataset_name) / f"ref_logps_{tag}.pt"


def reference_logps(cfg, dataset_name, rows, split, tok, ref_model):
    """Cached (logp_chosen, logp_rejected) under the shared reference, aligned with rows."""
    path = _ref_cache_path(cfg, dataset_name, split)
    if path.exists():
        return torch.load(path)
    print(f"  computing reference log-probs for {split} ({len(rows)} rows)", flush=True)
    out = torch.zeros(len(rows), 2)
    bs = cfg.get("ref_batch_size", 8)
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = rows[i: i + bs]
            seqs = [encode_pair(tok, r["prompt"], r["chosen"], cfg) for r in chunk] + \
                   [encode_pair(tok, r["prompt"], r["rejected"], cfg) for r in chunk]
            ids, att, rm = collate(seqs, tok.pad_token_id)
            lp, _ = sequence_logps(ref_model, ids.to(device()), att.to(device()), rm.to(device()))
            out[i: i + len(chunk), 0] = lp[: len(chunk)].cpu()
            out[i: i + len(chunk), 1] = lp[len(chunk):].cpu()
    torch.save(out, path)
    return out


def reference_logps_alone(cfg, dataset_name, rows, split, tok):
    """reference_logps for evaluation: the reference is loaded only when the cache is missing
    and is freed before the caller loads the policy, since at 8B the two do not fit together."""
    path = _ref_cache_path(cfg, dataset_name, split)
    if path.exists():
        return torch.load(path)
    ref_model = load_frozen(str(reference_dir(cfg, dataset_name) / "model"), cfg)
    out = reference_logps(cfg, dataset_name, rows, split, tok, ref_model)
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()
    return out


class BARPDPO(BaseMethod):
    name = "barp"

    def __init__(self, params, output_dir=None, cfg=None):
        super().__init__(params, output_dir)
        self.cfg = cfg

    # --- bias index -------------------------------------------------------------------
    mask = None   # declared-attribute mask (d,), set in run()

    @staticmethod
    def attr_dim(data):
        names = data["meta"].get("attribute_names")
        return len(names) if names else 1

    def declared_mask(self, data):
        names, decl = data["meta"].get("attribute_names"), self.params.get("declared_attributes")
        if not names or not decl:
            return None
        unknown = set(decl) - set(names)
        if unknown:
            raise ValueError(f"declared_attributes {unknown} not in {names}")
        return torch.tensor([1.0 if n in decl else 0.0 for n in names])

    def n_theta(self, data):
        n = self.params.get("n_bias_params", 0)
        if n == "per_annotator":
            return data["meta"]["n_annotators"]
        if n == "per_class":
            return len(data["meta"]["class_names"])
        return int(n)

    def row_index(self, rows, data, seed):
        n = self.n_theta(data)
        if n <= 1:
            return np.zeros(len(rows), dtype=np.int64)
        if self.params.get("n_bias_params") == "per_class":
            return np.array([r["annotator_class"] for r in rows], dtype=np.int64)
        if self.params.get("shuffle_annotator_ids"):
            return np.random.default_rng(seed).integers(n, size=len(rows))
        return np.array([r["annotator"] for r in rows], dtype=np.int64)

    @staticmethod
    def gdiff(rows, batch_idx, dev):
        """(batch, d) attribute differences g_w - g_l; scalar attributes give d = 1."""
        diffs = [np.atleast_1d(np.asarray(rows[i]["g_chosen"], dtype=np.float32)
                               - np.asarray(rows[i]["g_rejected"], dtype=np.float32)) for i in batch_idx]
        return torch.tensor(np.stack(diffs), dtype=torch.float32, device=dev)

    def extra_logit(self, rows, batch_idx, theta, idx_arr, dev):
        """Bias term for a batch: theta[idx] . (g_w - g_l); subclasses (rdpo) override."""
        gd = self.gdiff(rows, batch_idx, dev)
        if theta is None:
            return torch.zeros(gd.shape[0], device=dev), gd
        if self.mask is not None:
            gd = gd * self.mask.to(dev)
        k = torch.tensor(idx_arr[batch_idx], dtype=torch.long, device=dev)
        return (theta[k] * gd).sum(-1), gd

    needs_reference_model = False

    def log_ratio_margin(self, policy, tok, rows, bidx, ref_lp, dev):
        """[(log pi - log ref)(y_w) - (log pi - log ref)(y_l)] for a batch (before beta)."""
        cfg, bs = self.cfg, len(bidx)
        seqs = [encode_pair(tok, rows[i]["prompt"], rows[i]["chosen"], cfg) for i in bidx] + \
               [encode_pair(tok, rows[i]["prompt"], rows[i]["rejected"], cfg) for i in bidx]
        ids, att, rm = collate(seqs, tok.pad_token_id)
        lp, _ = sequence_logps(policy, ids.to(dev), att.to(dev), rm.to(dev))
        ref = ref_lp[bidx].to(dev)
        return (lp[:bs] - ref[:, 0]) - (lp[bs:] - ref[:, 1])

    # --- training -----------------------------------------------------------------------
    def run(self, data, seed, dataset_name="multipref"):
        cfg, out = self.cfg, self.output_dir / f"seed{seed}"
        if (out / "done.json").exists():
            print(f"  checkpoint exists at {out}, skipping training")
            return json.load(open(out / "done.json"))
        ref_dir = reference_dir(cfg, dataset_name) / "model"
        tok = load_tokenizer(str(ref_dir))
        policy = load_policy(str(ref_dir), cfg, trainable=True)
        lora = cfg.get("adapter", "none") != "none"
        ref_model = None if lora else load_frozen(str(ref_dir), cfg)
        rows = data["train"]
        ref_lp = reference_logps(cfg, dataset_name, rows, "train", tok, policy if lora else ref_model)
        if not lora and not self.needs_reference_model:
            del ref_model
            ref_model = None
            torch.cuda.empty_cache()
        self.ref_model = ref_model

        torch.manual_seed(seed)
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        n_theta = self.n_theta(data)
        idx_arr = self.row_index(rows, data, seed)
        dev = device()
        d = self.attr_dim(data)
        self.mask = self.declared_mask(data)
        # shared_mean: theta_k = theta_bar + eps_k with theta_bar its own parameter, so the
        # mean bias receives the full gradient instead of 1/m of it per annotator.
        shared_mean = bool(self.params.get("shared_mean")) and n_theta > 1
        # class_level: theta_k = theta_bar + delta_{class(k)} + eps_k; needs per-annotator eps
        # with real (unshuffled) ids, since delta is indexed through the annotator's class.
        class_level = bool(self.params.get("class_level")) and shared_mean
        if class_level and (self.params.get("n_bias_params") != "per_annotator" or self.params.get("shuffle_annotator_ids")):
            raise ValueError("class_level needs n_bias_params: per_annotator and real annotator ids")
        # theta_init: 0 (default) or "offline" = the corpus-level log-odds that the g=1
        # side wins on cross-group pairs (data/<name>/meta.json: offline_theta_pooled).
        init = torch.zeros(d)
        if self.params.get("theta_init") == "offline":
            off = np.atleast_1d(np.array(data["meta"]["offline_theta_pooled"], dtype=object))
            init = torch.tensor([float(v) if v is not None else 0.0 for v in off], dtype=torch.float32)
        theta_bar = torch.nn.Parameter(init.clone().view(1, d).to(dev)) if shared_mean else None
        eps_init = torch.zeros(d) if shared_mean else init
        eps = torch.nn.Parameter(eps_init.view(1, d).repeat(n_theta, 1).to(dev)) if n_theta > 0 else None
        delta = cls_of_ann = None
        if class_level:
            delta = torch.nn.Parameter(torch.zeros(len(data["meta"]["class_names"]), d, device=dev))
            cls_of_ann = torch.tensor(data["meta"]["annotator_class"], dtype=torch.long, device=dev)

        def compose():
            if eps is None:
                return None
            if not shared_mean:
                return eps
            t = theta_bar + eps
            return t + delta[cls_of_ann] if class_level else t

        theta = compose()
        params = [p for p in policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
        theta_params = [q for q in (theta_bar, delta, eps) if q is not None]
        opt_theta = torch.optim.Adam(theta_params, lr=cfg["theta_lr"]) if theta_params else None
        beta, bs, accum, total = cfg["beta"], cfg["batch_size"], cfg["grad_accum"], cfg["steps"]
        lam = cfg.get("theta_penalty_lambda", 0.0)

        # Each update uses the next batch_size * grad_accum pairs of the shuffled order. How they are
        # split into forward passes changes speed and memory only: the loss is the sum over the
        # step's pairs divided by their number, so every pair weighs the same however it is grouped.
        # train_micro_batch sets pairs per forward pass (default batch_size, the original split);
        # sort_by_length groups similar lengths so less padding is computed.
        per_step = bs * accum
        micro = cfg.get("train_micro_batch", bs)
        row_len = [len(r["prompt"]) + max(len(r["chosen"]), len(r["rejected"])) for r in rows] \
            if cfg.get("sort_by_length") else None
        policy.train()
        log, t0, ptr = [], time.time(), 0
        for step in range(total):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, cfg["warmup_frac"], cfg["lr"])
            acc_loss = acc_margin = acc_correct = 0.0
            step_idx = [order[(ptr + j) % len(order)] for j in range(per_step)]
            ptr += per_step
            if row_len is not None:
                step_idx.sort(key=lambda i: -row_len[i])           # longest first: peak memory early
            for c in range(0, per_step, micro):
                # rebuilt per micro-batch: the class gather saves its index for backward,
                # and that graph is freed by the first backward of the window
                theta = compose()
                bidx = step_idx[c: c + micro]
                u = beta * self.log_ratio_margin(policy, tok, rows, bidx, ref_lp, dev)
                b, gd = self.extra_logit(rows, bidx, theta, idx_arr, dev)
                loss = -F.logsigmoid(u + b).sum() / per_step
                if theta is not None and lam > 0:
                    loss = loss + lam * theta.mean() ** 2 * len(bidx) / per_step
                loss.backward()
                acc_loss += loss.item()
                acc_margin += u.sum().item() / per_step
                acc_correct += (u > 0).float().sum().item() / per_step
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if opt_theta is not None:
                opt_theta.step()
                opt_theta.zero_grad(set_to_none=True)
            rec = {"step": step, "loss": acc_loss, "margin": acc_margin, "acc": acc_correct}
            theta = compose()
            if theta is not None:
                rec["theta_mean"] = theta.mean().item()
                if d > 1:
                    rec["theta_mean_vec"] = theta.mean(0).tolist()
                if shared_mean:
                    rec["theta_bar"] = theta_bar.item() if d == 1 else theta_bar.view(-1).tolist()
                rec["theta_std"] = theta.std().item() if n_theta > 1 else 0.0
            log.append(rec)
            if step % 20 == 0:
                print(f"  step {step}/{total} loss {acc_loss:.4f} margin {acc_margin:.3f} acc {acc_correct:.3f} "
                      f"theta_mean {rec.get('theta_mean', 0):.3f} {(time.time()-t0)/(step+1):.1f}s/step "
                      f"mem {torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0:.1f}GB", flush=True)

        out.mkdir(parents=True, exist_ok=True)
        save_model = policy.merge_and_unload() if lora else policy
        save_model.to(torch.bfloat16).save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
        json.dump(log, open(out / "training_log.json", "w"))
        done = {"train_loss_last100": float(np.mean([r["loss"] for r in log[-100:]])),
                "train_acc_last100": float(np.mean([r["acc"] for r in log[-100:]])),
                "seconds_per_step": (time.time() - t0) / total,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0}
        if theta is not None:
            th = theta.detach().cpu()
            th = th.view(-1).tolist() if d == 1 else th.tolist()   # d=1 keeps the old flat format
            if shared_mean:
                done["theta_bar"] = theta_bar.item() if d == 1 else theta_bar.view(-1).tolist()
            if delta is not None:
                done["theta_class_delta"] = delta.detach().cpu().tolist()
            json.dump(th, open(out / "theta.json", "w"))
            done.update({"theta_mean": float(np.mean(th)), "theta_std": float(np.std(th)) if n_theta > 1 else 0.0})
            if d > 1:
                done["theta_mean_vec"] = np.mean(np.array(th), axis=0).tolist()
        json.dump(done, open(out / "done.json", "w"))
        del policy, opt, save_model      # the merged LoRA model would otherwise hold 16 GB at 8B
        gc.collect()
        torch.cuda.empty_cache()
        return done
