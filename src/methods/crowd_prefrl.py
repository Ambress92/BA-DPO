"""Crowd-PrefRL-style reliability weighting (Chhan et al. 2024, `chhan2024crowd`, adapted
to DPO): downweights whichever annotator is currently the LEAST reliable (highest loss),
the mirror image of Group-DRO. Discounts noisy annotators instead of protecting them, and
like Group-DRO uses no bias parameter and no g_chosen/g_rejected -- a heterogeneity
baseline contrasted against BARP-DPO's declared-attribute bias correction.
"""
import torch

from methods.group_dro_dpo import GroupWeightedDPO


class CrowdPrefRL(GroupWeightedDPO):
    def _update_weight(self, q, present, group_loss, eta):
        q[present] *= torch.exp(-eta * group_loss)
        q.clamp_(min=1e-4)
        q *= (len(q) / q.sum())
