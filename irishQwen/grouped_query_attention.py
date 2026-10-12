import math
import torch
import torch.nn.functional as F
from collections import namedtuple
from .rms_norm import RMSNorm


class GroupedQueryAttention:
    """
    Qwen3-style attention. Three changes vs. our GPT-2 MultiHeadAttention:

    1) Grouped-Query Attention (GQA): H query heads but only H_kv key/value heads.
       Each K/V head is shared by a group of G = H / H_kv query heads, which
       shrinks the KV cache by a factor of G.
    2) QK-Norm: an RMSNorm on every query and key head (over Dh) before RoPE.
       Keeps attention logits from blowing up during training.
    3) RoPE instead of learned positional embeddings, applied to Q and K only.

    Also: no biases, and head_dim is its own setting (H * Dh need not equal C).
    """
    Grad_Info = namedtuple(
        "Grad_Info",
        [
            "x",                      # (B, T, C)
            "qr", "kr",               # (B, H, T, Dh), (B, H_kv, T, Dh)  after norm + RoPE
            "kr_rep", "vh_rep",       # (B, H, T, Dh)  K/V repeated to match query heads
            "attn",                   # (B, H, T, T)
            "ctx",                    # (B, T, H*Dh)
            "key_pad_mask",           # (B, T) bool or None
            "start_pos",              # int
        ],
    )

    def __init__(self, model_dim, n_heads, n_kv_heads, head_dim, rope, device="cpu", causal=True):
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.C = model_dim
        self.H = n_heads
        self.H_kv = n_kv_heads
        self.G = n_heads // n_kv_heads      # query heads per K/V head
        self.Dh = head_dim
        self.device = device
        self.causal = causal
        self.rope = rope                    # shared RotaryEmbedding

        scale = math.sqrt(1.0 / self.C)

        # QK-Norm: one gamma of size Dh, shared across heads

    def _split_heads(self, X, n_heads):
        pass

    def _merge_heads(self, Xh):
        pass

    def _repeat_kv(self, Xh):
        pass

    def _sum_kv_groups(self, dXh):
        pass

    def forward(self, X, key_pad_mask=None, start_pos=0):
        B, T, _ = X.shape

        # QK-Norm, then rotate by position

        self.cache = GroupedQueryAttention.Grad_Info(
            x=X,
            qr=Qr, kr=Kr,
            kr_rep=Kr_rep, vh_rep=Vh_rep,
            attn=attn,
            ctx=ctx,
            key_pad_mask=key_pad_mask,
            start_pos=start_pos,
        )

        return Y

    def backward(self, dY):

        # ---- Output projection ----

        # ---- attn @ V ----

        # ---- Softmax ----

        # ---- Q @ K^T ----

        # ---- Un-repeat K/V (sum gradients over each group) ----

        # ---- RoPE, then QK-Norm ----


        # ---- Q/K/V projections ----

        return dX

    def update(self, lr):
        pass

    def load_from_qwen3_tensors(self, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm):
        """
        Expected incoming shapes (HF nn.Linear is (out, in), same as our X @ W.T, so no transposes):
          q_proj: (H*Dh, C)   k_proj, v_proj: (H_kv*Dh, C)   o_proj: (C, H*Dh)
          q_norm, k_norm: (Dh,)
        """
        expected = {
            "q_proj": (q_proj, (self.H * self.Dh, self.C)),
            "k_proj": (k_proj, (self.H_kv * self.Dh, self.C)),
            "v_proj": (v_proj, (self.H_kv * self.Dh, self.C)),
            "o_proj": (o_proj, (self.C, self.H * self.Dh)),
        }
        for name, (tensor, shape) in expected.items():
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")

        self.Wq = q_proj.to(self.device).contiguous()
        self.Wk = k_proj.to(self.device).contiguous()
        self.Wv = v_proj.to(self.device).contiguous()
        self.Wo = o_proj.to(self.device).contiguous()
        self.q_norm.gamma = q_norm.to(self.device).contiguous()
        self.k_norm.gamma = k_norm.to(self.device).contiguous()

    def forward_kv(self, x_new: torch.Tensor, kv_cache, start_pos: int):
        # During training, forward handles the whole sequence and saves what backward needs.
        # When generating, we keep the old keys and values so we only compute the new tokens.
        # Same weights, just reusing work! The first call handles the whole prompt.
        """
        Run attention for T new tokens (shape (1, T, C)) whose first token sits at
        absolute position `start_pos`.
          Prefill: the whole prompt at once (T = prompt length, start_pos = 0)
          Decode:  one token at a time       (T = 1, start_pos = tokens so far)
        """
        T = x_new.shape[1]

        # RoPE needs the tokens' real positions, not 0!

        # Cache the already-rotated keys, so old keys never need re-rotating.

        # Causal mask: new token i (at position start_pos + i) may see keys 0 ... start_pos + i.
        # During decode (T = 1) this masks nothing -- the cache only holds the past.

        return ctx @ self.Wo.T
