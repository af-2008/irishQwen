from .rms_norm import RMSNorm
from .swiglu import SwiGLU
from .qwen_moe import QwenSparseMoE
from .grouped_query_attention import GroupedQueryAttention

class QwenTransformer:
    attention_class = GroupedQueryAttention   # QwenTransformerWithKVCache swaps in a cached version

    def __init__(
        self,
        model_dim,
        rope,
        ffn_dim=3072,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        n_experts=0,
        top_k=2,
        device='cpu',
    ):
        """
        Qwen3 decoder block. Same Pre-Norm + residual skeleton as our GPT-2 Transformer:
          X -> RMSNorm -> GQA (QK-Norm + RoPE) -> +residual -> RMSNorm -> SwiGLU -> +residual

        model_dim: C (embedding size)
        rope: shared RotaryEmbedding
        n_experts: 0 = dense SwiGLU FFN (Qwen3), > 0 = sparse MoE FFN (Qwen3-MoE)
        """
        self.device = device

        self.LN1 = RMSNorm(model_dim, device)
        self.SA = self.attention_class(
            model_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, rope=rope, device=device
        )
        self.LN2 = RMSNorm(model_dim, device)
        if n_experts > 0:
            self.FFN = QwenSparseMoE(model_dim, ffn_dim, n_experts=n_experts, top_k=top_k, device=device)
        else:
            self.FFN = SwiGLU(model_dim, ffn_dim, device)

    def forward(self, X, key_pad_mask=None):
        resid = X

        X = self.LN1.forward(X)
        X = self.SA.forward(X, key_pad_mask=key_pad_mask)

        X = X + resid
        resid = X

        X = self.LN2.forward(X)
        X = self.FFN.forward(X)

        return X + resid

    def backward(self, dY):
        d_resid = dY

        dY = self.FFN.backward(dY)
        dY = self.LN2.backward(dY)

        dY = dY + d_resid

        d_resid = dY

        dY = self.SA.backward(dY)
        dY = self.LN1.backward(dY)

        return dY + d_resid

    def update(self, lr):
        self.LN1.update(lr)
        self.SA.update(lr)
        self.LN2.update(lr)
        self.FFN.update(lr)

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

        if isinstance(self.FFN, QwenSparseMoE):
            self.FFN.load_from_qwen3_tensors(block_state["mlp.gate.weight"], block_state["mlp.experts"])
        else:
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

        if isinstance(self.FFN, QwenSparseMoE):
            block_state["mlp.gate.weight"] = self.FFN.router.W.cpu()
            block_state["mlp.experts"] = [
                {"gate_proj": e.gate.W.cpu(), "up_proj": e.up.W.cpu(), "down_proj": e.down.W.cpu()}
                for e in self.FFN.experts
            ]
        else:
            block_state["mlp.gate_proj.weight"] = self.FFN.gate.W.cpu()
            block_state["mlp.up_proj.weight"] = self.FFN.up.W.cpu()
            block_state["mlp.down_proj.weight"] = self.FFN.down.W.cpu()

        return block_state
