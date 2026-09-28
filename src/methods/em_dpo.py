"""EM-DPO and MinMax-DPO (Chidambaram, Seetharaman and Syrgkanis 2024, arXiv 2405.15065): the
heterogeneity baseline built for latent annotator types inside DPO. It uses the annotator ids and
never the attribute.

EM-DPO keeps K policies, one per latent type, and a soft assignment gamma[i, k] of every
annotator i to type k. Starting from a random balanced assignment it alternates
  M-step: policy k is trained by DPO on votes drawn with probability gamma[annotator, k];
  E-step: gamma[i, k] is proportional to eta_k * prod_j sigmoid(u_k(vote j of i)) over the half of
          annotator i's training votes that the round's policies did not train on, with u_k the
          DPO margin of policy k and eta = mean gamma.
MinMax-DPO, the affine-ensemble variant the authors' code runs (find_weightsv2.py): policy j's
answers are scored by every type's implicit reward, L[i, j] = mean over prompts of
log pi_i(y_j) - log ref(y_j); the regret of type i when policy j answers is L[i, i] - L[i, j]; the
mixture weights over the K policies solve a zero-sum game (min over policies of the max regret
over types, plus a zero row) by optimistic multiplicative weights.

The output is a mixture: each prompt is answered by policy k with probability w_k, so every
readout is the w-weighted mean of the K type policies' readouts. The results row carries the
MinMax mixture at the top level, the eta-weighted (population) mixture under em_mixture and each
type policy's readouts under per_type. heldout_own_theta is EM-DPO's own vote prediction: each
annotator's votes predicted by their posterior mixture of type policies, the counterpart of the
BARP arms' per-annotator theta.

Departures from the authors' code:
- The E-step scores each annotator's votes that the type policies did not train on. Each
  annotator's training votes are split at random into two halves; in a round the policies train on
  one half and the type probabilities are computed on the other, and the halves swap every round.
  Scoring the training votes themselves (a first version of this code) let each policy explain its
  own annotators best, so a random start never moved (names_v3 seed 42: the same assignment after
  every round, 42% of annotators matched to their class). The authors avoid a random start instead,
  by starting from clusters computed beforehand.
- `em_init: agreement` starts from clusters of annotators who agree on the comparisons they share
  (agreement_clusters), computed from the votes alone. With the random start and 4 rounds the types
  only partly formed (names_v3 seed 42: 43% matched for three rounds, 53% after the
  fourth, class B never separated); the agreement clusters start at 60% with class B mostly apart.
- Each round trains the type policies from the reference for `round_steps` steps (the authors'
  code also restarts from the reference each round). After the last round the K policies are
  trained on all votes with the final type probabilities for `steps` steps, like every other arm,
  and those are the policies evaluated.
- The M-step samples votes in proportion to the type probabilities instead of multiplying the loss
  by them, so every type policy sees the same number of votes per step.
- The MinMax generations are the evaluation generations.
"""
import gc
import itertools
import json
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import collate, cosine_lr, device, encode_pair, load_frozen, load_policy, load_tokenizer, sequence_logps
from methods.barp_dpo import reference_logps_alone
from methods.base import BaseMethod
from methods.reference import reference_dir


def mix(results, w):
    """Weighted mean of K readout dicts, leaf by leaf; a non-numeric leaf is kept when all agree."""
    x0 = results[0]
    if isinstance(x0, dict):
        return {k: mix([r.get(k) if isinstance(r, dict) else None for r in results], w) for k in x0}
    if isinstance(x0, list) and all(isinstance(r, list) and len(r) == len(x0) for r in results):
        return [mix([r[i] for r in results], w) for i in range(len(x0))]
    if all(isinstance(r, (int, float)) and not isinstance(r, bool) for r in results):
        return float(sum(wi * r for wi, r in zip(w, results)))
    return x0 if all(r == x0 for r in results) else None


def minmax_weights(R, T=100000):
    """Optimistic multiplicative weights on the zero-sum game with payoff R (rows: the adversary's
    type, columns: the policies), as in the authors' code. Returns the time-averaged policy mix."""
    n, m = R.shape
    lr = 5 * np.sqrt(1 / T)
    x, y = np.ones(n) / n, np.ones(m) / m
    lx_prev, ly_prev = R @ y, R.T @ x
    ysum = np.zeros(m)
    for _ in range(T):
        lx, ly = R @ y, R.T @ x
        x = x * np.exp(2 * lr * lx - lr * lx_prev)
        x /= x.sum()
        y = y * np.exp(-2 * lr * ly + lr * ly_prev)
        y /= y.sum()
        lx_prev, ly_prev = lx, ly
        ysum += y
    return ysum / T


def type_recovery(gamma, annotator_class):
    """Confusion of true class against most likely type, and the share of annotators on the
    diagonal under the best matching of types to classes."""
    cls, hard = np.asarray(annotator_class), gamma.argmax(1)
    C, K = int(cls.max()) + 1, gamma.shape[1]
    conf = np.zeros((C, K), dtype=int)
    np.add.at(conf, (cls, hard), 1)
    best = max(sum(conf[c, p[c]] for c in range(C)) for p in itertools.permutations(range(K), C)) if K >= C else None
    return {"confusion": conf.tolist(), "matched_share": None if best is None else best / len(cls)}


def agreement_clusters(rows, ann, m, K, rng):
    """Annotators grouped by how often they agree, from the votes alone (never the attribute).
    A[i, j] is the share of the comparisons judged by both i and j on which they chose the same
    response; the centred matrix is embedded by its top K-1 eigenvectors (scaled by the eigenvalues)
    and split by k-means, best of 50 restarts. On names_v3 this matches 60% of annotators to their
    class, against about 40% for a random start."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["comparison_id"], []).append((ann[i], r["chosen"]))
    agree, n = np.zeros((m, m)), np.zeros((m, m))
    for votes in by.values():
        for (a, ca), (b, cb) in itertools.combinations(votes, 2):
            agree[a, b] += ca == cb
            agree[b, a] += ca == cb
            n[a, b] += 1
            n[b, a] += 1
    A = np.where(n > 0, agree / np.maximum(n, 1), np.nan)
    S = np.nan_to_num(A - np.nanmean(A))
    np.fill_diagonal(S, 0.0)
    w, V = np.linalg.eigh(S)
    X = V[:, -(K - 1):] * w[-(K - 1):]
    best = None
    for _ in range(50):
        C = X[rng.choice(m, K, replace=False)]
        for _ in range(100):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


class EMMinMaxDPO(BaseMethod):
    def __init__(self, params, output_dir=None, cfg=None):
        super().__init__(params, output_dir)
        self.cfg = cfg

    def _m_step(self, src, dst, p_row, rows, ref_lp, tok, rng, n_steps, log, k, r):
        """DPO from src for n_steps (its own cosine schedule) on votes drawn in proportion to p_row."""
        cfg, dev = self.cfg, device()
        lora = cfg.get("adapter", "none") != "none"
        policy = load_policy(str(src), cfg, trainable=True)
        params = [p for p in policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
        beta, bs, accum = cfg["beta"], cfg["batch_size"], cfg["grad_accum"]
        draw = rng.choice(len(rows), size=n_steps * bs * accum, p=p_row / p_row.sum())
        policy.train()
        t0, ptr = time.time(), 0
        for step in range(n_steps):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, n_steps, cfg["warmup_frac"], cfg["lr"])
            acc_loss = acc_correct = 0.0
            for _ in range(accum):
                bidx = draw[ptr: ptr + bs].tolist()
                ptr += bs
                seqs = [encode_pair(tok, rows[i]["prompt"], rows[i]["chosen"], cfg) for i in bidx] + \
                       [encode_pair(tok, rows[i]["prompt"], rows[i]["rejected"], cfg) for i in bidx]
                ids, att, rm = collate(seqs, tok.pad_token_id)
                lp, _ = sequence_logps(policy, ids.to(dev), att.to(dev), rm.to(dev))
                ref = ref_lp[bidx].to(dev)
                u = beta * ((lp[:bs] - ref[:, 0]) - (lp[bs:] - ref[:, 1]))
                loss = -F.logsigmoid(u).mean() / accum
                loss.backward()
                acc_loss += loss.item()
                acc_correct += (u > 0).float().mean().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            log.append({"type": k, "round": r, "step": step, "loss": acc_loss, "acc": acc_correct})
            if step % 20 == 0:
                print(f"  round {r} type {k} step {step}/{n_steps} loss {acc_loss:.4f} acc {acc_correct:.3f} "
                      f"{(time.time()-t0)/(step+1):.1f}s/step "
                      f"mem {torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0:.1f}GB", flush=True)
        dst.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + "_new")
        save = policy.merge_and_unload() if lora else policy
        save.to(torch.bfloat16).save_pretrained(tmp)
        tok.save_pretrained(tmp)
        del policy, opt, save
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(dst)
        tmp.rename(dst)

    def _margins(self, model_dir, rows, idx, ref_lp, tok):
        """beta * DPO margin of a saved policy on rows[idx], from the cached reference log-probs."""
        cfg = self.cfg
        model = load_frozen(str(model_dir), cfg)
        u, bs = np.zeros(len(idx)), cfg.get("ref_batch_size", 8)
        with torch.no_grad():
            for b in range(0, len(idx), bs):
                sub = idx[b: b + bs]
                seqs = [encode_pair(tok, rows[i]["prompt"], rows[i]["chosen"], cfg) for i in sub] + \
                       [encode_pair(tok, rows[i]["prompt"], rows[i]["rejected"], cfg) for i in sub]
                ids, att, rm = collate(seqs, tok.pad_token_id)
                lp = sequence_logps(model, ids.to(device()), att.to(device()), rm.to(device()))[0].float().cpu()
                n, ref = len(sub), ref_lp[torch.as_tensor(sub)]
                u[b: b + n] = (cfg["beta"] * ((lp[:n] - ref[:, 0]) - (lp[n:] - ref[:, 1]))).numpy()
        del model
        torch.cuda.empty_cache()
        return u

    def run(self, data, seed, dataset_name="multipref"):
        cfg, out = self.cfg, self.output_dir / f"seed{seed}"
        if (out / "done.json").exists():
            print(f"  checkpoint exists at {out}, skipping training")
            return json.load(open(out / "done.json"))
        K, R = int(self.params.get("n_types", 3)), int(self.params.get("em_rounds", 4))
        round_steps = int(self.params.get("round_steps", 200))
        ref_dir = reference_dir(cfg, dataset_name) / "model"
        tok = load_tokenizer(str(ref_dir))
        rows, meta = data["train"], data["meta"]
        ref_lp = reference_logps_alone(cfg, dataset_name, rows, "train", tok)
        m = meta["n_annotators"]
        ann = np.array([r["annotator"] for r in rows])
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if self.params.get("em_init", "random") == "agreement":      # clusters of annotators who vote alike
            gamma = np.eye(K)[agreement_clusters(rows, ann, m, K, rng)]
        else:                                                         # random balanced start
            gamma = np.eye(K)[rng.permutation(np.arange(m) % K)]
        if "annotator_class" in meta:
            print(f"  EM start ({self.params.get('em_init', 'random')}): "
                  f"{json.dumps(type_recovery(gamma, meta['annotator_class']))}", flush=True)
        # each annotator's votes split at random into two halves (fold 0 and 1)
        fold = np.zeros(len(rows), dtype=np.int64)
        for a in range(m):
            idx = np.flatnonzero(ann == a)
            rng.shuffle(idx)
            fold[idx[1::2]] = 1
        log, rounds, t0 = [], [], time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        for r in range(R):
            train_half = (fold == r % 2).astype(float)
            score_idx = np.flatnonzero(fold != r % 2)                 # votes the round's policies never see
            for k in range(K):
                # the floor keeps a type that lost every annotator trainable (it then sees all votes)
                self._m_step(ref_dir, out / "round" / f"type{k}", (gamma[ann, k] + 1e-6) * train_half,
                             rows, ref_lp, tok, rng, round_steps, log, k, r)
            loglik = np.zeros((m, K))
            for k in range(K):
                u = self._margins(out / "round" / f"type{k}", rows, score_idx, ref_lp, tok)
                loglik[:, k] = np.bincount(ann[score_idx], weights=-np.logaddexp(0.0, -u), minlength=m)
            logpost = np.log(gamma.mean(0) + 1e-12) + loglik
            logpost -= logpost.max(1, keepdims=True)
            gamma = np.exp(logpost) / np.exp(logpost).sum(1, keepdims=True)
            rec = {"round": r, "eta": gamma.mean(0).tolist()}
            if "annotator_class" in meta:
                rec["type_recovery"] = type_recovery(gamma, meta["annotator_class"])
            rounds.append(rec)
            print(f"  EM round {r}: {json.dumps(rec)}", flush=True)
        shutil.rmtree(out / "round", ignore_errors=True)
        for k in range(K):                                            # final type policies: all votes, full length
            self._m_step(ref_dir, out / f"type{k}" / "model", gamma[ann, k] + 1e-6, rows, ref_lp, tok, rng,
                         cfg["steps"], log, k, "final")
        out.mkdir(parents=True, exist_ok=True)
        json.dump(log, open(out / "training_log.json", "w"))
        done = {"gamma": np.round(gamma, 4).tolist(), "eta": gamma.mean(0).tolist(), "em_rounds": rounds,
                "train_loss_last100": float(np.mean([x["loss"] for x in log[-100:]])),
                "seconds_total": time.time() - t0,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0}
        json.dump(done, open(out / "done.json", "w"))
        return done

    def _minmax(self, out, K, dataset_name):
        cfg = self.cfg
        gens = [json.load(open(out / f"type{j}" / "generations.json")) for j in range(K)]
        tok = load_tokenizer(str(out / "type0" / "model"))
        dirs = [out / f"type{i}" / "model" for i in range(K)] + [reference_dir(cfg, dataset_name) / "model"]
        S, bs = np.zeros((K + 1, K)), cfg.get("ref_batch_size", 8)
        for i, d in enumerate(dirs):                      # S[i, j]: mean log-prob of policy j's answers under model i
            model = load_frozen(str(d), cfg)
            with torch.no_grad():
                for j in range(K):
                    tot = 0.0
                    for b in range(0, len(gens[j]), bs):
                        ids, att, rm = collate([encode_pair(tok, x["prompt"], x["response"], cfg) for x in gens[j][b: b + bs]],
                                               tok.pad_token_id)
                        tot += sequence_logps(model, ids.to(device()), att.to(device()), rm.to(device()))[0].sum().item()
                    S[i, j] = tot / len(gens[j])
            del model
            torch.cuda.empty_cache()
        L = S[:K] - S[K]
        regret = np.zeros((K + 1, K))
        regret[1:] = np.diag(L)[:, None] - L
        w = minmax_weights(regret)
        json.dump({"S": S.tolist(), "L": L.tolist(), "regret": regret.tolist(), "weights": w.tolist()},
                  open(out / "minmax.json", "w"))
        return w

    def evaluate(self, data, seed, dataset_name):
        from evaluation import evaluate_checkpoint
        from names_readouts import accuracies, policy_margin
        cfg, out = self.cfg, self.output_dir / f"seed{seed}"
        st = json.load(open(out / "done.json"))
        gamma, eta = np.array(st["gamma"]), np.array(st["eta"])
        K = len(eta)
        per_type = []
        for k in range(K):                                # cached per type, so a crash does not redo the rest
            path = out / f"type{k}" / "eval.json"
            if not path.exists():
                json.dump(evaluate_checkpoint(out / f"type{k}" / "model", data, cfg, dataset_name, params={}), open(path, "w"))
            per_type.append(json.load(open(path)))
        w = self._minmax(out, K, dataset_name)
        res = mix(per_type, w)
        res.update({"minmax_weights": w.tolist(), "em_mixture": mix(per_type, eta), "per_type": per_type})
        meta, rows = data["meta"], data["heldout"]
        if meta.get("attribute_names"):
            ann = np.array([r["annotator"] for r in rows])
            prob = sum(gamma[ann, k] / (1 + np.exp(-policy_margin(out / f"type{k}" / "model", rows, cfg, dataset_name)))
                       for k in range(K))
            logit = np.log(prob + 1e-12) - np.log(1 - prob + 1e-12)
            zeros = np.zeros((meta["n_annotators"], len(meta["attribute_names"])))
            own = accuracies(logit, rows, zeros, list(range(len(meta["class_names"]))))
            res["heldout_own_theta"] = own
            res["em_mixture"]["heldout_own_theta"] = own
        return res
