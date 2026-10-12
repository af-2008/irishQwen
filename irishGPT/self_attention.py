import math
import torch
import torch.nn.functional as F
from collections import namedtuple

class SelfAttention:
    Grad_Info = namedtuple('Grad_Info', [
        'x',
        'q', 'k', 'v',                    # (B, T, Dh)
        'scores',                          # (B, T, T)
        'attn',                            # (B, T, T)
        'attn_v',                          # (B, T, Dh)  = attn @ v
        'key_pad_mask',                    # (B, T) or None
    ])

    def __init__(self, model_dim, device='cpu', causal=True):
        self.model_dim = model_dim
        self.device = device

        # ---- Attention weights (Xavier init, appropriate for attention) ----
        scale_in = math.sqrt(1.0 / self.model_dim)
        self.Wq = torch.randn(self.model_dim, self.model_dim, device=self.device) * scale_in
        self.Wk = torch.randn(self.model_dim, self.model_dim, device=self.device) * scale_in
        self.Wv = torch.randn(self.model_dim, self.model_dim, device=self.device) * scale_in
        self.bq = torch.zeros(self.model_dim, device=self.device)
        self.bk = torch.zeros(self.model_dim, device=self.device)
        self.bv = torch.zeros(self.model_dim, device=self.device)

        self.Wo = torch.randn(self.model_dim, self.model_dim, device=self.device) * scale_in
        self.bo = torch.zeros(self.model_dim, device=self.device)

        self.cache = None
        self.causal = causal

    def forward(self, X, key_pad_mask=None):
        B, T, C = X.shape

        # ---- Self-Attention (computed from normalized input) ----
        Q = X @ self.Wq.T + self.bq
        K = X @ self.Wk.T + self.bk
        V = X @ self.Wv.T + self.bv

        scores = (Q @ K.transpose(1, 2)) / math.sqrt(self.model_dim)

        if self.causal:
            tril = torch.tril(torch.ones(T, T, device=X.device, dtype=torch.bool))
            scores = scores.masked_fill(~tril, float('-inf'))
        if key_pad_mask is not None:
            mask_k = key_pad_mask.unsqueeze(1).expand(B, T, T)
            scores = scores.masked_fill(mask_k, float('-inf'))

        attn = F.softmax(scores, dim=-1)   # (B, T, T)
        attn_v = attn @ V                  # (B, T, Dh)

        y_ctx = attn_v @ self.Wo.T + self.bo   # (B, T, C)

        self.cache = SelfAttention.Grad_Info(
            x=X, q=Q, k=K, v=V, scores=scores, attn=attn, attn_v=attn_v, key_pad_mask=key_pad_mask
        )

        return y_ctx

    def backward(self, dY):
        B, T, C = dY.shape
        dy_ctx = dY

        # ---- Output projection backward ----
        self.dWo = dy_ctx.reshape(-1, C).T @ self.cache.attn_v.reshape(-1, self.model_dim)
        self.dbo = dy_ctx.sum(dim=(0, 1))
        dattn_v  = dy_ctx @ self.Wo          # (B, T, Dh)

        # ---- Attention backward ----
        dv    = self.cache.attn.transpose(1, 2) @ dattn_v             # (B, T, Dh)
        dattn = dattn_v @ self.cache.v.transpose(1, 2)                # (B, T, T)

        # Softmax backward
        tmp     = (dattn * self.cache.attn).sum(dim=-1, keepdim=True) # (B, T, 1)
        dscores = (dattn - tmp) * self.cache.attn                     # (B, T, T)

        if self.causal:
            tril = torch.tril(torch.ones(T, T, device=dY.device, dtype=torch.bool))
            dscores = dscores.masked_fill(~tril, 0.0)
        if self.cache.key_pad_mask is not None:
            mask_k = self.cache.key_pad_mask.unsqueeze(1).expand(B, T, T)
            dscores = dscores.masked_fill(mask_k, 0.0)

        factor = 1.0 / math.sqrt(self.model_dim)
        dqk = dscores * factor                                   # (B, T, T)

        dq = dqk @ self.cache.k                                       # (B, T, Dh)
        dk = dqk.transpose(1, 2) @ self.cache.q                       # (B, T, Dh)

        # ---- Q/K/V parameter gradients (w.r.t. X_norm) ----
        self.dWq = dq.reshape(-1, self.model_dim).T @ self.cache.x.reshape(-1, self.model_dim)
        self.dbq = dq.sum(dim=(0, 1))

        self.dWk = dk.reshape(-1, self.model_dim).T @ self.cache.x.reshape(-1, self.model_dim)
        self.dbk = dk.sum(dim=(0, 1))

        self.dWv = dv.reshape(-1, self.model_dim).T @ self.cache.x.reshape(-1, self.model_dim)
        self.dbv = dv.sum(dim=(0, 1))

        # Gradient w.r.t. X_norm from Q, K, V branches
        dX = dq @ self.Wq + dk @ self.Wk + dv @ self.Wv    # (B, T, C)

        return dX

    def update(self, lr):
        # SGD update for attention weights
        self.Wq -= lr * self.dWq; self.bq -= lr * self.dbq
        self.Wk -= lr * self.dWk; self.bk -= lr * self.dbk
        self.Wv -= lr * self.dWv; self.bv -= lr * self.dbv
        self.Wo -= lr * self.dWo; self.bo -= lr * self.dbo