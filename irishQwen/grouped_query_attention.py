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
        self.Wq = torch.randn(self.H * self.Dh, self.C, device=self.device) * scale
        self.Wk = torch.randn(self.H_kv * self.Dh, self.C, device=self.device) * scale
        self.Wv = torch.randn(self.H_kv * self.Dh, self.C, device=self.device) * scale
        self.Wo = torch.randn(self.C, self.H * self.Dh, device=self.device) * math.sqrt(1.0 / (self.H * self.Dh))

        # QK-Norm: one gamma of size Dh, shared across heads
        self.q_norm = RMSNorm(self.Dh, device=device)
        self.k_norm = RMSNorm(self.Dh, device=device)

    def _split_heads(self, X, n_heads):
        B, T, _ = X.shape
        return X.view(B, T, n_heads, self.Dh).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, Xh):
        B, H, T, Dh = Xh.shape
        return Xh.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)

    def _repeat_kv(self, Xh):
        # (B, H_kv, T, Dh) -> (B, H, T, Dh): kv head j serves query heads j*G ... j*G + G-1
        return Xh.repeat_interleave(self.G, dim=1)

    def _sum_kv_groups(self, dXh):
        # Backward of _repeat_kv: each kv head gets the summed gradient of its G copies
        B, H, T, Dh = dXh.shape
        return dXh.view(B, self.H_kv, self.G, T, Dh).sum(dim=2)

    def forward(self, X, key_pad_mask=None, start_pos=0):
        B, T, _ = X.shape

        Qh = self._split_heads(X @ self.Wq.T, self.H)       # (B, H, T, Dh)
        Kh = self._split_heads(X @ self.Wk.T, self.H_kv)    # (B, H_kv, T, Dh)
        Vh = self._split_heads(X @ self.Wv.T, self.H_kv)    # (B, H_kv, T, Dh)

        # QK-Norm, then rotate by position
        Qr = self.rope.forward(self.q_norm.forward(Qh), start_pos)
        Kr = self.rope.forward(self.k_norm.forward(Kh), start_pos)

        Kr_rep = self._repeat_kv(Kr)
        Vh_rep = self._repeat_kv(Vh)

        scale = 1.0 / math.sqrt(self.Dh)
        scores = (Qr @ Kr_rep.transpose(-1, -2)) * scale    # (B, H, T, T)

        if self.causal:
            tril = torch.tril(torch.ones(T, T, device=X.device, dtype=torch.bool))
            scores = scores.masked_fill(~tril.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_pad_mask is not None:
            mask_k = key_pad_mask.unsqueeze(1).unsqueeze(1).expand(B, self.H, T, T)
            scores = scores.masked_fill(mask_k, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        ctx = self._merge_heads(attn @ Vh_rep)              # (B, T, H*Dh)
        Y = ctx @ self.Wo.T                                 # (B, T, C)

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
        B, T, C = dY.shape
        cache = self.cache

        # ---- Output projection ----
        self.dWo = dY.reshape(-1, C).T @ cache.ctx.reshape(-1, self.H * self.Dh)
        dctx = dY @ self.Wo

        # ---- attn @ V ----
        dctx_heads = self._split_heads(dctx, self.H)
        dattn = dctx_heads @ cache.vh_rep.transpose(-1, -2)
        dVh_rep = cache.attn.transpose(-1, -2) @ dctx_heads

        # ---- Softmax ----
        tmp = (dattn * cache.attn).sum(dim=-1, keepdim=True)
        dscores = (dattn - tmp) * cache.attn

        if self.causal:
            tril = torch.tril(torch.ones(T, T, device=dY.device, dtype=torch.bool))
            dscores = dscores.masked_fill(~tril.unsqueeze(0).unsqueeze(0), 0.0)

        if cache.key_pad_mask is not None:
            mask_k = cache.key_pad_mask.unsqueeze(1).unsqueeze(1).expand(B, self.H, T, T)
            dscores = dscores.masked_fill(mask_k, 0.0)

        # ---- Q @ K^T ----
        scale = 1.0 / math.sqrt(self.Dh)
        dQr = (dscores @ cache.kr_rep) * scale
        dKr_rep = (dscores.transpose(-1, -2) @ cache.qr) * scale

        # ---- Un-repeat K/V (sum gradients over each group) ----
        dKr = self._sum_kv_groups(dKr_rep)
        dVh = self._sum_kv_groups(dVh_rep)

        # ---- RoPE, then QK-Norm ----
        dQh = self.q_norm.backward(self.rope.backward(dQr, cache.start_pos))
        dKh = self.k_norm.backward(self.rope.backward(dKr, cache.start_pos))

        dQ = self._merge_heads(dQh)     # (B, T, H*Dh)
        dK = self._merge_heads(dKh)     # (B, T, H_kv*Dh)
        dV = self._merge_heads(dVh)     # (B, T, H_kv*Dh)

        # ---- Q/K/V projections ----
        x_flat = cache.x.reshape(-1, self.C)
        self.dWq = dQ.reshape(-1, self.H * self.Dh).T @ x_flat
        self.dWk = dK.reshape(-1, self.H_kv * self.Dh).T @ x_flat
        self.dWv = dV.reshape(-1, self.H_kv * self.Dh).T @ x_flat

        dX = dQ @ self.Wq + dK @ self.Wk + dV @ self.Wv
        return dX

    def update(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv
        self.Wo -= lr * self.dWo
        self.q_norm.update(lr)
        self.k_norm.update(lr)

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
