"""Model loading, tokenisation and sequence log-probabilities shared by every method."""
import contextlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SHARED_DIR = Path(__file__).resolve().parent.parent / "experiments" / "shared"


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cap_gpu_memory(cfg):
    """Hard per-process cap so this project can never exceed its share of the shared GPU."""
    if torch.cuda.is_available() and cfg.get("gpu_memory_cap_gb"):
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["gpu_memory_cap_gb"] / total), 0)


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # generation; training uses explicit masks
    return tok


def load_policy(name, cfg, trainable=True):
    """Full FT: fp32 master weights. LoRA: frozen bf16 base + fp32 adapters (peft)."""
    dev = device()
    adapter = cfg.get("adapter", "none")
    if adapter == "none":
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    else:
        from peft import LoraConfig, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        if trainable:
            lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                              target_modules=cfg.get("lora_target", "all-linear"), task_type="CAUSAL_LM")
            model = get_peft_model(model, lcfg)
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
    if trainable:
        model.config.use_cache = False       # training never reads the attention cache
    if trainable and cfg.get("gradient_checkpointing", True):
        # Recompute each layer's activations in the backward pass instead of keeping them: less
        # memory, more compute. Gradients are the same either way (difference 0 measured on
        # Qwen2.5-0.5B), so a config may switch it off when memory allows.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model.to(dev)


def load_frozen(name, cfg):
    """Reference / judge-side copy in bf16, no grads."""
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@contextlib.contextmanager
def adapters_disabled(model):
    """LoRA models: the base with adapters off is the reference policy."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def encode_pair(tok, prompt, response, cfg):
    """Token ids for chat(prompt) + response + eos, with a response mask.
    Prompt is left-truncated to max_prompt_length; total right-truncated to max_length."""
    p_text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                     add_generation_prompt=True, tokenize=False)
    p_ids = tok(p_text, add_special_tokens=False)["input_ids"][-cfg["max_prompt_length"]:]
    r_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + r_ids)[: cfg["max_length"]]
    mask = ([0] * len(p_ids) + [1] * len(r_ids))[: cfg["max_length"]]
    return ids, mask


def collate(seqs, pad_id):
    """Right-pad a list of (ids, mask) into tensors. Returns input_ids, attention, resp_mask."""
    L = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    rm = torch.zeros((len(seqs), L), dtype=torch.float32)
    for i, (s, m) in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
        rm[i, : len(m)] = torch.tensor(m, dtype=torch.float32)
    return ids, att, rm


def _chunk_logps(lm_head, h, targets, chunk):
    """Per-position log p(target) from hidden states, computed in position chunks so the
    full [B, L, V] logits never exist at once. Each chunk is checkpointed for backward."""
    from torch.utils.checkpoint import checkpoint

    def f(hc, tc):
        logits = lm_head(hc).float()
        return torch.gather(logits, -1, tc.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)

    outs = []
    for i in range(0, h.shape[1], chunk):
        hc, tc = h[:, i: i + chunk], targets[:, i: i + chunk]
        outs.append(checkpoint(f, hc, tc, use_reentrant=False) if torch.is_grad_enabled() else f(hc, tc))
    return torch.cat(outs, dim=1)


def _base_and_head(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head


def token_logps(model, ids, att, resp_mask, autocast=True, chunk=256):
    """Per-token log p over response positions (0 elsewhere), shape [B, L-1], and the mask."""
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (autocast and ids.is_cuda) else contextlib.nullcontext()
    base, head = _base_and_head(model)
    with ctx:
        h = base(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        logp = _chunk_logps(head, h, ids[:, 1:], chunk)
    mask = resp_mask[:, 1:]
    return logp * mask, mask


def sequence_logps(model, ids, att, resp_mask, autocast=True):
    """Sum of log p(token) over response tokens, per sequence, and the token counts."""
    logp, mask = token_logps(model, ids, att, resp_mask, autocast)
    return logp.sum(-1), mask.sum(-1)


def cosine_lr(step, total, warmup_frac, base):
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1 + math.cos(math.pi * prog))
