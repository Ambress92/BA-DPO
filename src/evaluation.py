"""One evaluation path for every arm.

evaluate_checkpoint(model_dir, data, cfg, dataset_name) returns:
  attr_rate            marginal attribute rate on generations
  attr_rate_std        length-standardized rate (reweighted onto the reference's length bins)
  mean_tokens          mean generated tokens
  sig_p_woman          names corpora: probability of a woman's name at the signature
                       position, over the pool, with the reference's answer body
  sig_p_black          the same for a Black-coded name; sig_p_cell splits it by cell
  heldout_own_theta    names corpora: vote prediction with the arm's own theta, by class;
                       heldout_true_theta is the planted ceiling, heldout_policy_only theta=0
  reward_raw           judge score on raw generations
  reward_inv           judge score after the attribute-invariant transform (None for length)
  heldout_acc_*        accuracy of sign(u) against observed labels: all / same_group / cross_group
Generations are saved next to the checkpoint so the judge can be re-run without regenerating.
"""
import json
from pathlib import Path

import numpy as np
import torch

from attributes import get_attribute, strip_for_judge
from common import SHARED_DIR, collate, device, encode_pair, load_frozen, load_tokenizer, sequence_logps
from methods.barp_dpo import reference_logps_alone
from methods.reference import reference_dir

# A pair is cross-group when the two sides differ in the attribute by more than this.
# Binary attributes give |1-0| = 1 and |0-0| = 0, so their behaviour is unchanged.
CROSS_MARGIN = 0.25


def generate(model_dir, prompts, cfg, out_path):
    if Path(out_path).exists():
        return json.load(open(out_path))
    tok = load_tokenizer(str(model_dir))
    model = load_frozen(str(model_dir), cfg)
    gens, bs = [], cfg.get("gen_batch_size", 16)
    torch.manual_seed(cfg.get("gen_seed", 0))
    for i in range(0, len(prompts), bs):
        chunk = prompts[i: i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         tokenize=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device())
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=cfg["gen_max_new_tokens"], do_sample=True,
                                 temperature=cfg["gen_temperature"], top_p=cfg["gen_top_p"],
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for p, ids in zip(chunk, new):
            n = int((ids != tok.pad_token_id).sum())
            gens.append({"prompt": p, "response": tok.decode(ids, skip_special_tokens=True), "tokens": n})
    del model
    torch.cuda.empty_cache()
    json.dump(gens, open(out_path, "w"))
    return gens


def _num(x):
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x.tolist()


def attribute_rates(gens, attr_name, ref_gens, cfg=None):
    """Marginal rate and a rate reweighted onto the reference generations' token-count bins.
    Vector attributes (names) return one value per attribute, as lists."""
    toks = np.array([x["tokens"] for x in gens], dtype=float)
    if attr_name == "length":
        return float(toks.mean()), float(toks.mean())
    g = get_attribute(attr_name, cfg)
    flags = np.array([g(x["response"], "", x["prompt"])[0] for x in gens], dtype=float)   # (n,) or (n, d)
    ref_toks = np.array([x["tokens"] for x in ref_gens], dtype=float)
    edges = np.quantile(ref_toks, [0.2, 0.4, 0.6, 0.8])
    ref_bin = np.digitize(ref_toks, edges)
    w = np.bincount(ref_bin, minlength=5) / len(ref_toks)
    b = np.digitize(toks, edges)
    per_bin = np.stack([flags[b == k].mean(0) if (b == k).any() else flags.mean(0) for k in range(5)])
    std = np.tensordot(w, per_bin, axes=(0, 0))
    return _num(flags.mean(0)), _num(std)


def _choice_pick(response, cand):
    """Which candidate a choice-prompt answer recommends: the one mentioned first, by full
    name, first name or 'Candidate k'. None if neither is mentioned. A rough reading;
    the paper reports the decided fraction next to the rate."""
    import re
    pos = []
    for k in (1, 2):
        full = cand[f"name_{k}"]
        pats = [re.escape(full), r"\b" + re.escape(full.split()[0]) + r"\b", rf"[Cc]andidate\s*{k}\b"]
        hits = [m.start() for p in pats for m in [re.search(p, response)] if m]
        pos.append(min(hits) if hits else None)
    if pos[0] is None and pos[1] is None:
        return None
    if pos[1] is None or (pos[0] is not None and pos[0] <= pos[1]):
        return cand["cell_1"]
    return cand["cell_2"]


def names_rates(sign_gens, choice_gens, data):
    """Extra readouts for the names corpus. On sign prompts: the fraction signed at all and
    the rate of each name cell. On choice prompts: for each attribute, among prompts where
    the two candidates differ on it, the share of recommendations going to the candidate
    carrying it (0.5 = no preference)."""
    from attributes import CELL_VECTOR, signature_cell
    cells = [signature_cell(x["response"]) for x in sign_gens]
    out = {"signed_frac": float(np.mean([c is not None for c in cells])),
           "cell_rates": {cell: float(np.mean([c == cell for c in cells])) for cell in CELL_VECTOR}}
    cand = data["meta"].get("choice_candidates", [])
    names = data["meta"]["attribute_names"]
    picks = [_choice_pick(x["response"], cm) for x, cm in zip(choice_gens, cand)]
    out["choice_decided_frac"] = float(np.mean([p is not None for p in picks])) if picks else None
    pref = []
    for j in range(len(names)):
        wins = []
        for p, cm in zip(picks, cand):
            v1, v2 = CELL_VECTOR[cm["cell_1"]][j], CELL_VECTOR[cm["cell_2"]][j]
            if p is None or v1 == v2:
                continue
            wins.append(CELL_VECTOR[p][j] == 1)
        pref.append(float(np.mean(wins)) if wins else None)
    out["choice_pref"] = pref
    return out


_JUDGE = {}


def judge_scores(gens, cfg, transform=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = cfg["judge"]
    if name not in _JUDGE:
        tok = AutoTokenizer.from_pretrained(name)
        rm = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.bfloat16, num_labels=1).to(device())
        rm.eval()
        _JUDGE[name] = (tok, rm)
    tok, rm = _JUDGE[name]
    scores, bs = [], cfg.get("judge_batch_size", 8)
    for i in range(0, len(gens), bs):
        chunk = gens[i: i + bs]
        convs = [[{"role": "user", "content": x["prompt"]},
                  {"role": "assistant", "content": transform(x["response"]) if transform else x["response"]}]
                 for x in chunk]
        texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["judge_max_length"]).to(device())
        with torch.no_grad():
            scores += rm(**enc).logits[:, 0].float().cpu().tolist()
    return scores


def free_judge():
    import gc
    _JUDGE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def heldout_accuracy(model_dir, data, cfg, dataset_name):
    rows = data["heldout"]
    tok = load_tokenizer(str(model_dir))
    ref = reference_logps_alone(cfg, dataset_name, rows, "heldout", tok)
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
    correct = (u > 0).numpy()
    gc = np.array([np.atleast_1d(r["g_chosen"]) for r in rows], dtype=float)
    gr = np.array([np.atleast_1d(r["g_rejected"]) for r in rows], dtype=float)
    diff = np.abs(gc - gr) > CROSS_MARGIN            # (n, d)
    cross = diff.any(1)
    out = {"heldout_acc_all": float(correct.mean()),
           "heldout_acc_same_group": float(correct[~cross].mean()) if (~cross).any() else None,
           "heldout_acc_cross_group": float(correct[cross].mean()) if cross.any() else None}
    names = data["meta"].get("attribute_names")
    if diff.shape[1] > 1 and names:
        out["heldout_acc_per_attr"] = {
            n: {"same": float(correct[~diff[:, j]].mean()) if (~diff[:, j]).any() else None,
                "cross": float(correct[diff[:, j]].mean()) if diff[:, j].any() else None}
            for j, n in enumerate(names)}
    return out


def kl_to_reference(model_dir, gens, cfg, dataset_name):
    """Monte Carlo KL(pi || ref) on the policy's own generations: the mean over response
    tokens of log pi(y|x) - log ref(y|x) with y ~ pi. Reported per token and per response."""
    tok = load_tokenizer(str(model_dir))
    bs = cfg.get("ref_batch_size", 8)
    batches = [collate([encode_pair(tok, x["prompt"], x["response"], cfg) for x in gens[i: i + bs]], tok.pad_token_id)
               for i in range(0, len(gens), bs)]
    # one model on the GPU at a time: at 8B the policy and the reference do not fit together
    lps = {}
    for key, path in (("ref", reference_dir(cfg, dataset_name) / "model"), ("pi", model_dir)):
        model = load_frozen(str(path), cfg)
        with torch.no_grad():
            lps[key] = [tuple(t.cpu() for t in sequence_logps(model, ids.to(device()), att.to(device()), rm.to(device())))
                        for ids, att, rm in batches]
        del model
        torch.cuda.empty_cache()
    tot = sum((lp_m - lp_r).sum().item() for (lp_m, _), (lp_r, _) in zip(lps["pi"], lps["ref"]))
    ntok = sum(n.sum().item() for _, n in lps["pi"])
    return {"kl_per_token": tot / max(ntok, 1.0), "kl_per_response": tot / max(len(gens), 1)}


def evaluate_checkpoint(model_dir, data, cfg, dataset_name, is_reference=False, params=None):
    model_dir = Path(model_dir)
    attr = cfg["attribute"]
    ref_dir = reference_dir(cfg, dataset_name)
    ref_gens = generate(ref_dir / "model", data["eval_prompts"], cfg, ref_dir / "generations.json")
    gens = ref_gens if is_reference else generate(model_dir, data["eval_prompts"], cfg, model_dir.parent / "generations.json")
    types = data["meta"].get("eval_prompt_types")
    extra = {}
    if types:   # names: attribute rates on the sign prompts only; choice prompts read separately
        sign = [i for i, t in enumerate(types) if t == "sign"]
        choice = [i for i, t in enumerate(types) if t == "choice"]
        extra = names_rates([gens[i] for i in sign], [gens[i] for i in choice], data)
        rate, rate_std = attribute_rates([gens[i] for i in sign], attr, [ref_gens[i] for i in sign], cfg)
    else:
        rate, rate_std = attribute_rates(gens, attr, ref_gens, cfg)
    res = {"attr_rate": rate, "attr_rate_std": rate_std,
           "mean_tokens": float(np.mean([x["tokens"] for x in gens])), **extra}
    raw = judge_scores(gens, cfg)
    res["reward_raw"] = float(np.mean(raw))
    inv_fn = strip_for_judge(attr, "") is not None
    if inv_fn:
        res["reward_inv"] = float(np.mean(judge_scores(gens, cfg, transform=lambda y: strip_for_judge(attr, y))))
    else:
        res["reward_inv"] = None
    free_judge()
    if not is_reference:
        res.update(heldout_accuracy(model_dir, data, cfg, dataset_name))
        res.update(kl_to_reference(model_dir, gens, cfg, dataset_name))
    else:
        res.update({"kl_per_token": 0.0, "kl_per_response": 0.0})
    if data["meta"].get("attribute_names") and types:
        # names corpora: the signature read from probabilities and the vote prediction with the
        # arm's own theta. Rows written by an earlier version of the pipeline lack these keys
        # (scripts/analysis/signature_probs.py and heldout_with_theta.py add them).
        from names_readouts import heldout_with_theta, signature_probabilities, strip_sig
        sign = [i for i, t in enumerate(types) if t == "sign"]
        res.update(signature_probabilities(model_dir, data, cfg,
                                           [data["eval_prompts"][i] for i in sign],
                                           [strip_sig(ref_gens[i]["response"]) for i in sign]))
        if not is_reference:
            res.update(heldout_with_theta(model_dir, data, cfg, dataset_name, params or {}))
    return res
