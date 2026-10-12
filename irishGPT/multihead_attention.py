import math
import torch
import torch.nn.functional as F
from collections import namedtuple


class MultiHeadAttention:
    Grad_Info = namedtuple(
        "Grad_Info",
        [
            "x",
            "q", "k", "v",            # (B, T, C)
            "qh", "kh", "vh",         # (B, H, T, Dh)
            "scores",                 # (B, H, T, T)
            "attn",                   # (B, H, T, T)
            "ctx", "ctx_heads",       # (B, T, C), (B, H, T, Dh)
            "key_pad_mask",           # (B, T) bool or None
        ],
    )

    def __init__(self, model_dim, n_heads, device="cpu", causal=True):
        self.C = model_dim
        self.H = n_heads
        self.Dh = model_dim // n_heads
        self.device = device
        self.causal = causal

        scale = math.sqrt(1.0 / self.C)
        self.Wq = torch.randn(self.C, self.C, device=self.device) * scale
        self.Wk = torch.randn(self.C, self.C, device=self.device) * scale
        self.Wv = torch.randn(self.C, self.C, device=self.device) * scale
        self.bq = torch.zeros(self.C, device=self.device)
        self.bk = torch.zeros(self.C, device=self.device)
        self.bv = torch.zeros(self.C, device=self.device)

        self.Wo = torch.randn(self.C, self.C, device=self.device) * scale
        self.bo = torch.zeros(self.C, device=self.device)

    def _split_heads(self, X):
        B, T, C = X.shape
        return X.view(B, T, self.H, self.Dh).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, Xh):
        B, H, T, Dh = Xh.shape
        return Xh.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)

    def forward(self, X, key_pad_mask=None):
        B, T, _ = X.shape

        Q = X @ self.Wq.T + self.bq
        K = X @ self.Wk.T + self.bk
        V = X @ self.Wv.T + self.bv

        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scale = 1.0 / math.sqrt(self.Dh)
        scores = (Qh @ Kh.transpose(-1, -2)) * scale

        if self.causal:
            tril = torch.tril(torch.ones(T, T, device=X.device, dtype=torch.bool))
            scores = scores.masked_fill(~tril.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_pad_mask is not None:
            mask_k = key_pad_mask.unsqueeze(1).unsqueeze(1).expand(B, self.H, T, T)
            scores = scores.masked_fill(mask_k, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        ctx_heads = attn @ Vh
        ctx = self._merge_heads(ctx_heads)
        Y = ctx @ self.Wo.T + self.bo

        self.cache = MultiHeadAttention.Grad_Info(
            x=X,
            q=Q, k=K, v=V,
            qh=Qh, kh=Kh, vh=Vh,
            scores=scores,
            attn=attn,
            ctx=ctx,
            ctx_heads=ctx_heads,
            key_pad_mask=key_pad_mask,
        )
        return Y

    def backward(self, dY):
        B, T, C = dY.shape

        self.dWo = dY.reshape(-1, C).T @ self.cache.ctx.reshape(-1, C)
        self.dbo = dY.sum(dim=(0, 1))
        dctx = dY @ self.Wo

        dctx_heads = self._split_heads(dctx)
        dattn = dctx_heads @ self.cache.vh.transpose(-1, -2)
        dVh = self.cache.attn.transpose(-1, -2) @ dctx_heads

        tmp = (dattn * self.cache.attn).sum(dim=-1, keepdim=True)
        dscores = (dattn - tmp) * self.cache.attn

        if self.causal:
            tril = torch.tril(torch.ones(T, T, device=dY.device, dtype=torch.bool))
            dscores = dscores.masked_fill(~tril.unsqueeze(0).unsqueeze(0), 0.0)

        if self.cache.key_pad_mask is not None:
            mask_k = self.cache.key_pad_mask.unsqueeze(1).unsqueeze(1).expand(B, self.H, T, T)
            dscores = dscores.masked_fill(mask_k, 0.0)

        scale = 1.0 / math.sqrt(self.Dh)
        dQh = (dscores @ self.cache.kh) * scale
        dKh = (dscores.transpose(-1, -2) @ self.cache.qh) * scale

        dQ = self._merge_heads(dQh)
        dK = self._merge_heads(dKh)
        dV = self._merge_heads(dVh)

        x_flat = self.cache.x.reshape(-1, self.C)
        self.dWq = dQ.reshape(-1, self.C).T @ x_flat
        self.dbq = dQ.sum(dim=(0, 1))
        self.dWk = dK.reshape(-1, self.C).T @ x_flat
        self.dbk = dK.sum(dim=(0, 1))
        self.dWv = dV.reshape(-1, self.C).T @ x_flat
        self.dbv = dV.sum(dim=(0, 1))

        dX = dQ @ self.Wq + dK @ self.Wk + dV @ self.Wv
        return dX

    def update(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv
        self.Wo -= lr * self.dWo
        self.bq -= lr * self.dbq
        self.bk -= lr * self.dbk
        self.bv -= lr * self.dbv
        self.bo -= lr * self.dbo

    def load_from_gpt2_tensors(self, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias):
        """
        Map GPT-2 block attention tensors into this module.
        Expected incoming shapes:
          c_attn_weight: (C, 3C), c_attn_bias: (3C,)
          c_proj_weight: (C, C), c_proj_bias: (C,)
        """
        if c_attn_weight.shape != (self.C, 3 * self.C):
            raise ValueError(f"c_attn_weight shape mismatch: expected {(self.C, 3 * self.C)}, got {tuple(c_attn_weight.shape)}")
        if c_attn_bias.shape != (3 * self.C,):
            raise ValueError(f"c_attn_bias shape mismatch: expected {(3 * self.C,)}, got {tuple(c_attn_bias.shape)}")
        if c_proj_weight.shape != (self.C, self.C):
            raise ValueError(f"c_proj_weight shape mismatch: expected {(self.C, self.C)}, got {tuple(c_proj_weight.shape)}")
        if c_proj_bias.shape != (self.C,):
            raise ValueError(f"c_proj_bias shape mismatch: expected {(self.C,)}, got {tuple(c_proj_bias.shape)}")

        wq, wk, wv = torch.chunk(c_attn_weight, 3, dim=1)
        bq, bk, bv = torch.chunk(c_attn_bias, 3, dim=0)

        # Our forward uses X @ W.T, GPT-2 uses X @ W, so transpose q/k/v slices.
        self.Wq = wq.T.to(self.device).contiguous()
        self.Wk = wk.T.to(self.device).contiguous()
        self.Wv = wv.T.to(self.device).contiguous()
        self.bq = bq.to(self.device).contiguous()
        self.bk = bk.to(self.device).contiguous()
        self.bv = bv.to(self.device).contiguous()

        # Same convention for output projection.
        self.Wo = c_proj_weight.T.to(self.device).contiguous()
        self.bo = c_proj_bias.to(self.device).contiguous()
