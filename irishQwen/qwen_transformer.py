from dataclasses import dataclass
import torch

from .rms_norm import RMSNorm
from .swiglu import SwiGLU
from .grouped_query_attention import GroupedQueryAttention

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
        pass

    def clear(self):
        pass


class QwenTransformer:
    attention_class = GroupedQueryAttention

    def __init__(
        self,
        model_dim,
        rope,
        ffn_dim=3072,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        device='cpu',
    ):
        """
        Qwen3 decoder block. Same Pre-Norm + residual skeleton as our GPT-2 Transformer:
          X -> RMSNorm -> GQA (QK-Norm + RoPE) -> +residual -> RMSNorm -> SwiGLU -> +residual

        model_dim: C (embedding size)
        rope: shared RotaryEmbedding
        """
        self.device = device

    def forward(self, X, key_pad_mask=None):
        pass

    def backward(self, dY):
        pass

    def update(self, lr):
        pass

    def load_from_qwen3_block(self, block_state):
        self.LN1.gamma = block_state["input_layernorm.weight"].to(self.device).contiguous()
        self.LN2.gamma = block_state["post_attention_layernorm.weight"].to(self.device).contiguous()

        self.SA.load_from_qwen3_tensors(
            block_state["self_attn.q_proj.weight"],
            block_state["self_attn.k_proj.weight"],
            block_state["self_attn.v_proj.weight"],
            block_state["self_attn.o_proj.weight"],
            block_state["self_attn.q_norm.weight"],
            block_state["self_attn.k_norm.weight"],
        )

        self.FFN.load_from_qwen3_tensors(
            block_state["mlp.gate_proj.weight"],
            block_state["mlp.up_proj.weight"],
            block_state["mlp.down_proj.weight"],
        )

    def save_to_qwen3_block(self):
        """Inverse of load_from_qwen3_block: this block's weights as an HF-style dict."""
        block_state = {
            "input_layernorm.weight": self.LN1.gamma.cpu(),
            "post_attention_layernorm.weight": self.LN2.gamma.cpu(),
            "self_attn.q_proj.weight": self.SA.Wq.cpu(),
            "self_attn.k_proj.weight": self.SA.Wk.cpu(),
            "self_attn.v_proj.weight": self.SA.Wv.cpu(),
            "self_attn.o_proj.weight": self.SA.Wo.cpu(),
            "self_attn.q_norm.weight": self.SA.q_norm.gamma.cpu(),
            "self_attn.k_norm.weight": self.SA.k_norm.gamma.cpu(),
        }

        block_state["mlp.gate_proj.weight"] = self.FFN.gate.W.cpu()
        block_state["mlp.up_proj.weight"] = self.FFN.up.W.cpu()
        block_state["mlp.down_proj.weight"] = self.FFN.down.W.cpu()

        return block_state

    def reset_kv_cache(self):
        self.kv_cache.clear()

    def forward_kv(self, x_new: torch.Tensor, start_pos: int):
        # During training, forward handles the whole sequence and saves what backward needs.
        # When generating, we keep the old keys and values so we only compute the new tokens.
        # Same weights, just reusing work! The first call handles the whole prompt.
        """
        Cached forward for generation (prefill chunk or one decode token).
        x_new shape: (1, T, C)
        """

        pass
