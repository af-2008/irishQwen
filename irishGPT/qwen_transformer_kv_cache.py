import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .grouped_query_attention import GroupedQueryAttention
from .qwen_transformer import QwenTransformer


@dataclass
class SimpleKVCache:
    """
    Tiny educational KV cache for one transformer block.

    We only support batch size = 1. Prefill appends the whole prompt at once,
    then each decode step appends one token.
    With GQA we only cache the H_kv key/value heads, not all H query heads:
      (1, n_kv_heads, cached_tokens, head_dim)
    For Qwen3-0.6B that is 8 heads instead of 16 -> half the memory.
    """

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k = k_new
            self.v = v_new
            return
        self.k = torch.cat([self.k, k_new], dim=2)
        self.v = torch.cat([self.v, v_new], dim=2)

    def clear(self):
        self.k = None
        self.v = None


class GroupedQueryAttentionWithKVCache(GroupedQueryAttention):
    """
    GroupedQueryAttention plus a cached inference path.
    forward/backward (full sequence, for training) are inherited unchanged.
    """

    def forward_step(self, x_new: torch.Tensor, kv_cache: SimpleKVCache, start_pos: int):
        """
        Run attention for T new tokens (shape (1, T, C)) whose first token sits at
        absolute position `start_pos`.
          Prefill: the whole prompt at once (T = prompt length, start_pos = 0)
          Decode:  one token at a time       (T = 1, start_pos = tokens so far)
        """
        T = x_new.shape[1]
        qh = self._split_heads(x_new @ self.Wq.T, self.H)        # (1, H, T, Dh)
        kh = self._split_heads(x_new @ self.Wk.T, self.H_kv)     # (1, H_kv, T, Dh)
        vh = self._split_heads(x_new @ self.Wv.T, self.H_kv)

        # RoPE needs the tokens' real positions, not 0!
        qr = self.rope.forward(self.q_norm.forward(qh), start_pos=start_pos)
        kr = self.rope.forward(self.k_norm.forward(kh), start_pos=start_pos)

        # Cache the already-rotated keys, so old keys never need re-rotating.
        kv_cache.append(kr, vh)

        k_all = self._repeat_kv(kv_cache.k)     # (1, H, start_pos + T, Dh)
        v_all = self._repeat_kv(kv_cache.v)

        scores = (qr @ k_all.transpose(-1, -2)) * (1.0 / math.sqrt(self.Dh))  # (1, H, T, start_pos + T)

        # Causal mask: new token i (at position start_pos + i) may see keys 0 ... start_pos + i.
        # During decode (T = 1) this masks nothing -- the cache only holds the past.
        q_pos = torch.arange(start_pos, start_pos + T, device=x_new.device).unsqueeze(1)
        k_pos = torch.arange(start_pos + T, device=x_new.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        ctx = self._merge_heads(attn @ v_all)
        return ctx @ self.Wo.T


class QwenTransformerWithKVCache(QwenTransformer):
    """
    Qwen3 block with per-block KV cache.

    Mirrors QwenTransformer:
      X -> RMSNorm -> GQA -> +residual -> RMSNorm -> SwiGLU/MoE -> +residual
    """

    attention_class = GroupedQueryAttentionWithKVCache

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache = SimpleKVCache()

    def reset_kv_cache(self):
        self.kv_cache.clear()

    def forward_step(self, x_new: torch.Tensor, start_pos: int):
        """
        Cached forward for generation (prefill chunk or one decode token).
        x_new shape: (1, T, C)
        """
        resid = x_new
        x = self.LN1.forward(x_new)
        x = self.SA.forward_step(x, kv_cache=self.kv_cache, start_pos=start_pos)
        x = x + resid

        resid = x
        x = self.LN2.forward(x)
        x = self.FFN.forward(x)
        return x + resid
