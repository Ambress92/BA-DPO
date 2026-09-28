from methods.base import BaseMethod
from methods.barp_dpo import BARPDPO
from methods.rdpo import RDPO
from methods.sampo import SamPO
from methods.group_dro_dpo import GroupDRODPO
from methods.crowd_prefrl import CrowdPrefRL
from methods.em_dpo import EMMinMaxDPO
from methods.fair_offset import FairOffsetDPO

# dpo / barp_pooled / barp_annotator / barp_shuffled are the same class; the config's
# params (n_bias_params, shuffle_annotator_ids) select the arm. reference is handled by
# the experiment script (methods/reference.py), sampo down-samples token log-ratios to equal length.
# group_dro_dpo / crowd_prefrl are heterogeneity baselines: annotator-group loss
# reweighting instead of an attribute-specific bias term (see group_dro_dpo.py).
METHOD_REGISTRY = {
    "dpo": BARPDPO,
    "barp_pooled": BARPDPO,
    "barp_annotator": BARPDPO,
    "barp_shuffled": BARPDPO,
    "barp_annotator_mean": BARPDPO,
    "barp_annotator_mean_warm": BARPDPO,
    "barp_pooled_warm": BARPDPO,
    # names corpus (vector attribute, annotator classes); params select the arm as above
    "barp_class": BARPDPO,
    "barp_class_mean": BARPDPO,
    "barp_pooled_gender_only": BARPDPO,
    "barp_class_gender_only": BARPDPO,
    "rdpo": RDPO,
    "rdpo_a005": RDPO,
    "sampo": SamPO,
    "group_dro_dpo": GroupDRODPO,
    "crowd_prefrl": CrowdPrefRL,
    # latent annotator types (Chidambaram et al. 2024): K type policies by EM, one MinMax mixture
    "em_minmax_dpo": EMMinMaxDPO,
    # DPO with a fixed fairness offset on a biased reference; the
    # config's offset selects the target rate
    "fair_offset": FairOffsetDPO,
    "fair_offset_t30": FairOffsetDPO,
    "fair_offset_t70": FairOffsetDPO,
    "fair_offset_cal_add": FairOffsetDPO,   # calibration reruns: additive and
    "fair_offset_cal_mul": FairOffsetDPO,   # proportional correction of the parity offset
    "fair_offset_fit": FairOffsetDPO,       # offsets from the fitted response line (seed 42): parity,
    "fair_offset_fit_t30": FairOffsetDPO,   # target 0.3 and target 0.7
    "fair_offset_fit_t70": FairOffsetDPO,
}


def get_method(name, params, output_dir=None, cfg=None):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name](params, output_dir=output_dir, cfg=cfg)
