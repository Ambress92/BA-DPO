"""R-DPO (Park et al. 2024): DPO with a length penalty. The regularised objective
max E[r] - alpha|y| - beta KL gives r = beta log(pi/ref) + alpha|y| + const, so the
Bradley-Terry logit is  u + alpha * (|y_w| - |y_l|)  (word counts here): when the chosen
response is longer, part of the preference is attributed to length and the update shrinks.
Same structure as BARP's u + theta (g_w - g_l). A minus sign here would make the policy
learn MORE from long winners."""
import torch

from methods.barp_dpo import BARPDPO


class RDPO(BARPDPO):
    def n_theta(self, data):
        return 0

    def extra_logit(self, rows, batch_idx, theta, idx_arr, dev):
        alpha = self.params["alpha"]
        dl = torch.tensor([len(rows[i]["chosen"].split()) - len(rows[i]["rejected"].split())
                           for i in batch_idx], dtype=torch.float32, device=dev)
        return alpha * dl, self.gdiff(rows, batch_idx, dev)
