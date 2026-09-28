"""Shared scaffold for annotator-group loss-reweighting baselines (Group-DRO, Crowd-PrefRL):
heterogeneity baselines from the branch of related work BARP-DPO is positioned against.
Neither uses a bias parameter or g_chosen/g_rejected at all -- they reweight the plain DPO
loss per annotator group instead of adding an attribute-specific term to the logit, which is
the whole point of the contrast with BARP-DPO.

Per-group weight q is updated once per optimiser step (after the full gradient-accumulation
window, the same cadence as BARP's theta): q[g] *= exp(+-eta * group_loss[g]), clamped and
renormalised to keep the mean weight at 1 so the effective loss scale stays comparable to
plain DPO. Subclasses differ only in the sign of the exponent (_update_weight).
"""
import gc
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import collate, cosine_lr, device, encode_pair, load_policy, load_tokenizer, sequence_logps
from methods.barp_dpo import reference_logps_alone
from methods.base import BaseMethod
from methods.reference import reference_dir


class GroupWeightedDPO(BaseMethod):
    eta = 0.5

    def __init__(self, params, output_dir=None, cfg=None):
        super().__init__(params, output_dir)
        self.cfg = cfg

    def n_groups(self, data):
        n = self.params.get("n_groups", "per_annotator")
        if n == "per_annotator":
            return data["meta"]["n_annotators"]
        if n == "per_class":                       # names corpus: groups = annotator classes
            return len(data["meta"]["class_names"])
        return max(1, int(n))

    def group_index(self, rows, data):
        n = self.n_groups(data)
        if n <= 1:
            return np.zeros(len(rows), dtype=np.int64)
        if self.params.get("n_groups") == "per_class":
            return np.array([r["annotator_class"] for r in rows], dtype=np.int64)
        return np.array([r["annotator"] for r in rows], dtype=np.int64)

    def _update_weight(self, q, present, group_loss, eta):
        raise NotImplementedError

    def run(self, data, seed, dataset_name="multipref"):
        cfg, out = self.cfg, self.output_dir / f"seed{seed}"
        if (out / "done.json").exists():
            print(f"  checkpoint exists at {out}, skipping training")
            return json.load(open(out / "done.json"))
        ref_dir = reference_dir(cfg, dataset_name) / "model"
        tok = load_tokenizer(str(ref_dir))
        rows = data["train"]
        # the cached reference log-probs first, so the reference and the policy never share the GPU
        ref_lp = reference_logps_alone(cfg, dataset_name, rows, "train", tok)
        policy = load_policy(str(ref_dir), cfg, trainable=True)
        lora = cfg.get("adapter", "none") != "none"

        torch.manual_seed(seed)
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        n_groups = self.n_groups(data)
        gidx = self.group_index(rows, data)
        dev = device()
        q = torch.full((n_groups,), 1.0, device=dev)
        eta = self.params.get("eta", self.eta)

        params = [p for p in policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
        beta, bs, accum, total = cfg["beta"], cfg["batch_size"], cfg["grad_accum"], cfg["steps"]

        policy.train()
        log, t0, ptr = [], time.time(), 0
        for step in range(total):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, cfg["warmup_frac"], cfg["lr"])
            acc_loss = acc_margin = acc_correct = 0.0
            step_losses, step_gidx = [], []
            for _ in range(accum):
                bidx = [order[(ptr + j) % len(order)] for j in range(bs)]
                ptr += bs
                seqs = [encode_pair(tok, rows[i]["prompt"], rows[i]["chosen"], cfg) for i in bidx] + \
                       [encode_pair(tok, rows[i]["prompt"], rows[i]["rejected"], cfg) for i in bidx]
                ids, att, rm = collate(seqs, tok.pad_token_id)
                lp, _ = sequence_logps(policy, ids.to(dev), att.to(dev), rm.to(dev))
                ref = ref_lp[bidx].to(dev)
                u = beta * ((lp[:bs] - ref[:, 0]) - (lp[bs:] - ref[:, 1]))
                per_ex_loss = -F.logsigmoid(u)
                bg = torch.tensor(gidx[bidx], dtype=torch.long, device=dev)
                w = q[bg]
                loss = (w * per_ex_loss).sum() / (w.sum() * accum)
                loss.backward()
                acc_loss += loss.item()
                acc_margin += u.mean().item() / accum
                acc_correct += (u > 0).float().mean().item() / accum
                step_losses.append(per_ex_loss.detach())
                step_gidx.append(bg)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                all_loss = torch.cat(step_losses)
                all_gidx = torch.cat(step_gidx)
                present = all_gidx.unique()
                group_loss = torch.stack([all_loss[all_gidx == g_].mean() for g_ in present])
                self._update_weight(q, present, group_loss, eta)
            log.append({"step": step, "loss": acc_loss, "margin": acc_margin, "acc": acc_correct,
                        "q_std": q.std().item()})
            if step % 20 == 0:
                print(f"  step {step}/{total} loss {acc_loss:.4f} margin {acc_margin:.3f} acc {acc_correct:.3f} "
                      f"q_std {q.std().item():.4f} {(time.time()-t0)/(step+1):.1f}s/step "
                      f"mem {torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0:.1f}GB", flush=True)

        out.mkdir(parents=True, exist_ok=True)
        save_model = policy.merge_and_unload() if lora else policy
        save_model.to(torch.bfloat16).save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
        json.dump(log, open(out / "training_log.json", "w"))
        json.dump(q.detach().cpu().tolist(), open(out / "q_weights.json", "w"))
        done = {"train_loss_last100": float(np.mean([r["loss"] for r in log[-100:]])),
                "train_acc_last100": float(np.mean([r["acc"] for r in log[-100:]])),
                "seconds_per_step": (time.time() - t0) / total,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
                "q_std": float(q.std().item())}
        json.dump(done, open(out / "done.json", "w"))
        del policy, opt, save_model
        gc.collect()
        torch.cuda.empty_cache()
        return done


class GroupDRODPO(GroupWeightedDPO):
    """Upweights whichever annotator group is currently doing worst (Sagawa et al. 2019,
    Group-DRO, adapted to DPO): robustness to disagreement, not bias correction."""
    def _update_weight(self, q, present, group_loss, eta):
        q[present] *= torch.exp(eta * group_loss)
        q.clamp_(min=1e-4)
        q *= (len(q) / q.sum())
