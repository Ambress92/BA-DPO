"""Readouts specific to the `names` corpora, used by evaluate_checkpoint and by the two
scripts that re-score checkpoints trained before these became part of the pipeline.

signature_probabilities: the model writes nothing. An answer body with its signature removed
gets the signature prefix appended, and every first name in the build's pool is scored as the
continuation: log P(name | prompt, body, prefix), summed over the name's tokens, normalised
over the pool and added up per cell. That gives, per prompt, the probability of a woman's name
and of a Black-coded name with no sampling and no unsigned answers. The prompt and body are
read once and every name starts from that cached state; the output layer is in fp32, since
bf16 logits of size 10-30 are rounded in steps of 1/16.

heldout_with_theta: the standard held-out accuracy scores sign(u) alone, i.e. what the policy
prefers. This one scores sign(u + theta_k . (g_c - g_r)) with the arm's own learned theta, so
it measures how well the arm predicts the votes each annotator class actually cast. Two
anchors: theta = 0 (the policy alone) and the planted theta_k (the ceiling).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attributes import CELL_VECTOR, FIRST_NAMES, _SIGNATURE
from common import (_base_and_head, collate, device, encode_pair, load_frozen, load_tokenizer,
                    sequence_logps)
from methods.reference import reference_dir

PREFIX = "\n\n— "


def name_list():
    """(first name, cell) for the pool this build activated in attributes.FIRST_NAMES."""
    return [(n, cell) for cell, ns in FIRST_NAMES.items() for n in ns]


def strip_sig(y):
    return _SIGNATURE.sub("", y).rstrip()


def encode_name_batch(tok, prompt, body, cfg):
    """One sequence per name: chat(prompt) + body + PREFIX + name, masked on the name tokens."""
    p_text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    p_ids = tok(p_text, add_special_tokens=False)["input_ids"][-cfg["max_prompt_length"]:]
    pre_ids = tok(body + PREFIX, add_special_tokens=False)["input_ids"]
    seqs = []
    for name, _ in name_list():
        full = tok(body + PREFIX + name, add_special_tokens=False)["input_ids"]
        k = 0
        while k < min(len(pre_ids), len(full)) and pre_ids[k] == full[k]:
            k += 1                                   # tokens shared with the prefix
        ids = (p_ids + full)[: cfg["max_length"]]
        mask = ([0] * (len(p_ids) + k) + [1] * (len(full) - k))[: cfg["max_length"]]
        seqs.append((ids, mask))
    return seqs


def _logits32(W, h):
    with torch.autocast(h.device.type, enabled=False):
        return F.linear(h.float(), W)


def _name_logps_cached(base, W, tok, seqs, dev):
    """Log P(name) for every name from one read of the shared context, or None when the names
    do not share one (a name fully truncated, or a different split point)."""
    starts = [m.index(1) if 1 in m else -1 for _, m in seqs]
    S = starts[0]
    if S <= 0 or any(x != S for x in starts) or any(ids[:S] != seqs[0][0][:S] for ids, _ in seqs):
        return None
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        o = base(input_ids=torch.tensor([seqs[0][0][:S]], device=dev), use_cache=True)
    first = F.log_softmax(_logits32(W, o.last_hidden_state[:, -1]), -1)[0]
    lp = first[torch.tensor([ids[S] for ids, _ in seqs], device=dev)].clone()
    multi = [i for i, (ids, _) in enumerate(seqs) if len(ids) - S >= 2]
    if multi:                                        # 2nd and 3rd tokens: one batched step
        M, B = max(len(seqs[i][0]) - S for i in multi), len(multi)
        inp = torch.full((B, M - 1), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((B, S + M - 1), dtype=torch.long)
        att[:, :S] = 1
        tgt = torch.zeros((B, M - 1), dtype=torch.long)
        keep = torch.zeros((B, M - 1))
        for b, i in enumerate(multi):
            ids, n = seqs[i][0], len(seqs[i][0]) - S
            inp[b, : n - 1] = torch.tensor(ids[S : S + n - 1])
            att[b, S : S + n - 1] = 1
            tgt[b, : n - 1] = torch.tensor(ids[S + 1 : S + n])
            keep[b, : n - 1] = 1
        past = o.past_key_values
        past.batch_repeat_interleave(B)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            o2 = base(input_ids=inp.to(dev), attention_mask=att.to(dev), past_key_values=past, use_cache=True)
        l2 = F.log_softmax(_logits32(W, o2.last_hidden_state), -1)
        rest = (l2.gather(-1, tgt.to(dev).unsqueeze(-1)).squeeze(-1) * keep.to(dev)).sum(-1)
        idx = torch.tensor(multi, device=dev)
        lp[idx] += rest
    return lp.float().cpu().numpy()


def _name_logps_full(base, W, tok, seqs, dev):
    """Fallback: the whole sequence re-read for each name, batches of 12."""
    out = []
    for i in range(0, len(seqs), 12):
        ids, att, rm = (t.to(dev) for t in collate(seqs[i: i + 12], tok.pad_token_id))
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            h = base(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        pos = rm[:, 1:].bool().nonzero()
        tgt = ids[:, 1:][pos[:, 0], pos[:, 1]]
        tl = F.log_softmax(_logits32(W, h[pos[:, 0], pos[:, 1]]), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        lp = torch.zeros(len(ids), device=dev)
        lp.index_add_(0, pos[:, 0], tl)
        out.append(lp.cpu())
    return torch.cat(out).numpy()


def cell_probs(model, tok, prompts, bodies, cfg):
    """Per-prompt P(woman), P(black) over the name pool, plus the per-name log-probs."""
    base, head = _base_and_head(model)
    W = head.weight.detach().float()
    dev = device()
    names = name_list()
    vec = np.array([CELL_VECTOR[c] for _, c in names], dtype=float)
    out = []
    with torch.no_grad():
        for prompt, body in zip(prompts, bodies):
            seqs = encode_name_batch(tok, prompt, body, cfg)
            lp = _name_logps_cached(base, W, tok, seqs, dev)
            if lp is None:
                lp = _name_logps_full(base, W, tok, seqs, dev)
            p = np.exp(lp - lp.max()); p /= p.sum()
            out.append({"p_woman": float(p @ vec[:, 0]), "p_black": float(p @ vec[:, 1]),
                        "p_cell": {c: float(p[[i for i, (_, cc) in enumerate(names) if cc == c]].sum()) for c in CELL_VECTOR},
                        "logp": lp.tolist()})
    return out


def summarise(per_prompt):
    return {"p_woman": float(np.mean([r["p_woman"] for r in per_prompt])),
            "p_black": float(np.mean([r["p_black"] for r in per_prompt])),
            "p_cell": {c: float(np.mean([r["p_cell"][c] for r in per_prompt])) for c in CELL_VECTOR}}


def signature_probabilities(model_dir, data, cfg, sign_prompts, ref_bodies):
    """Pipeline readout: probabilities at the signature position with the reference's answer
    body, which is the same text for every arm. The script also scores the arm's own body;
    on names_v3 the two differ by at most 0.005."""
    tok = load_tokenizer(str(model_dir))
    model = load_frozen(str(model_dir), cfg)
    try:
        s = summarise(cell_probs(model, tok, sign_prompts, ref_bodies, cfg))
    finally:
        del model
        torch.cuda.empty_cache()
    return {"sig_p_woman": s["p_woman"], "sig_p_black": s["p_black"], "sig_p_cell": s["p_cell"]}


def policy_margin(model_dir, rows, cfg, dataset_name, split="heldout"):
    from methods.barp_dpo import reference_logps_alone
    tok = load_tokenizer(str(model_dir))
    ref = reference_logps_alone(cfg, dataset_name, rows, split, tok)
    model = load_frozen(str(model_dir), cfg)
    u = torch.zeros(len(rows))
    bs = cfg.get("ref_batch_size", 8)
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = rows[i: i + bs]
            seqs = [encode_pair(tok, r["prompt"], r["chosen"], cfg) for r in chunk] + \
                   [encode_pair(tok, r["prompt"], r["rejected"], cfg) for r in chunk]
            ids, att, rm = collate(seqs, tok.pad_token_id)
            lp, _ = sequence_logps(model, ids.to(device()), att.to(device()), rm.to(device()))
            n = len(chunk)
            u[i: i + n] = cfg["beta"] * ((lp[:n].cpu() - ref[i: i + n, 0]) - (lp[n:].cpu() - ref[i: i + n, 1]))
    del model
    torch.cuda.empty_cache()
    return u.numpy()


def per_annotator_theta(method_dir, params, meta):
    """(n_annotators, d) theta as the arm would apply it to each row's annotator."""
    d = len(meta["attribute_names"])
    m = meta["n_annotators"]
    path = Path(method_dir) / "theta.json"
    if not path.exists():                      # dpo / baselines: no bias term
        return np.zeros((m, d))
    th = np.array(json.load(open(path)), dtype=float).reshape(-1, d)
    n = params.get("n_bias_params", 0)
    if n == "per_class":
        return th[np.array(meta["annotator_class"])]
    if n == "per_annotator":
        return th                              # shared_mean / class_level already composed
    return np.repeat(th[:1], m, axis=0)        # pooled


def accuracies(u, rows, theta_by_ann, classes):
    gd = np.array([np.subtract(r["g_chosen"], r["g_rejected"]) for r in rows], dtype=float)
    ann = np.array([r["annotator"] for r in rows])
    cls = np.array([r["annotator_class"] for r in rows])
    logit = u + (theta_by_ann[ann] * gd).sum(1)
    correct = logit > 0
    cross = (gd != 0).any(1)
    out = {"all": float(correct.mean()), "cross": float(correct[cross].mean()), "same": float(correct[~cross].mean())}
    for j in range(gd.shape[1]):
        xj = gd[:, j] != 0
        out[f"attr{j}_cross"] = float(correct[xj].mean())
        out[f"attr{j}_cross_by_class"] = [float(correct[xj & (cls == c)].mean()) for c in classes]
    return out


def heldout_with_theta(model_dir, data, cfg, dataset_name, params):
    """Pipeline readout: vote prediction with the arm's own theta, with theta = 0 and the
    planted theta_k as the two anchors."""
    meta, rows = data["meta"], data["heldout"]
    classes = list(range(len(meta["class_names"])))
    u = policy_margin(model_dir, rows, cfg, dataset_name)
    theta_true = np.array(meta["theta_true"], dtype=float)
    return {"heldout_policy_only": accuracies(u, rows, np.zeros_like(theta_true), classes),
            "heldout_own_theta": accuracies(u, rows, per_annotator_theta(Path(model_dir).parent, params, meta), classes),
            "heldout_true_theta": accuracies(u, rows, theta_true, classes)}
