"""DPO with a fairness offset: the BARP bias term with a fixed
vector b in place of learned parameters,  logit = u + b . (g_w - g_l).
Trained on pairs whose two responses differ only in the attribute, each pair in both orders,
the loss of the two orders is -log s(u + b) - log s(-u - b), minimised at u = -b: the policy
ends with log-odds for the attribute equal to its reference's minus b / beta. With a biased
policy as the reference, b = beta * (logit p_ref - logit t) moves its attribute rate from
p_ref to the target t. params: offset, one entry per attribute (data meta attribute_names)."""
import torch

from methods.barp_dpo import BARPDPO


class FairOffsetDPO(BARPDPO):
    def n_theta(self, data):
        return 0

    def extra_logit(self, rows, batch_idx, theta, idx_arr, dev):
        gd = self.gdiff(rows, batch_idx, dev)
        b = torch.tensor(self.params["offset"], dtype=torch.float32, device=dev).view(1, -1)
        if b.shape[1] != gd.shape[1]:
            raise ValueError(f"offset has {b.shape[1]} entries, the attribute has {gd.shape[1]}")
        return (b * gd).sum(-1), gd
