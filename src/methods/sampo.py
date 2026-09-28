"""SamPO (Lu et al. 2024): length-desensitised DPO. The per-token log-ratios
(log pi - log ref) of the longer response are randomly down-sampled to the shorter
response's token count before summing, so both sides contribute equally many tokens.
Needs token-level reference log-probs, so the reference model stays resident; with LoRA the
reference is the policy with its adapters switched off, so no second copy is loaded."""
import torch

from common import adapters_disabled, collate, encode_pair, token_logps
from methods.barp_dpo import BARPDPO


class SamPO(BARPDPO):
    needs_reference_model = True

    def n_theta(self, data):
        return 0

    def log_ratio_margin(self, policy, tok, rows, bidx, ref_lp, dev):
        cfg, bs = self.cfg, len(bidx)
        seqs = [encode_pair(tok, rows[i]["prompt"], rows[i]["chosen"], cfg) for i in bidx] + \
               [encode_pair(tok, rows[i]["prompt"], rows[i]["rejected"], cfg) for i in bidx]
        ids, att, rm = collate(seqs, tok.pad_token_id)
        ids, att, rm = ids.to(dev), att.to(dev), rm.to(dev)
        lp, mask = token_logps(policy, ids, att, rm)
        with torch.no_grad():
            if self.ref_model is None:                     # LoRA: BARPDPO.run loads no second copy
                with adapters_disabled(policy):
                    ref, _ = token_logps(policy, ids, att, rm)
            else:
                ref, _ = token_logps(self.ref_model, ids, att, rm)
        ratio = lp - ref                                   # per-token log-ratio, 0 off-response
        out = torch.zeros(bs, device=dev)
        for j in range(bs):
            pc, pr = mask[j].nonzero().squeeze(-1), mask[bs + j].nonzero().squeeze(-1)
            n = min(len(pc), len(pr))
            if n == 0:
                continue
            sc = pc[torch.randperm(len(pc), device=dev)[:n]]
            sr = pr[torch.randperm(len(pr), device=dev)[:n]]
            out[j] = ratio[j, sc].sum() - ratio[bs + j, sr].sum()
        return out
