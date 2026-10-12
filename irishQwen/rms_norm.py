import torch
from collections import namedtuple

class RMSNorm:
    """
    Root-Mean-Square LayerNorm (used by Llama, Qwen, Mistral, ...).

    Compared to LayerNorm:
      - No mean subtraction (we don't re-center, only re-scale)
      - No beta (bias), only gamma
    y = gamma * x / sqrt(mean(x^2) + eps)

    Works on any input shape (..., D). Qwen3 also uses this on each attention
    head separately (QK-Norm), where the input is (B, H, T, Dh).
    """
    Grad_Info = namedtuple('Grad_Info', [
        'x_hat',       # (..., D) normalized: X / rms
        'rms',         # (..., 1) root-mean-square (for backward)
    ])

    def __init__(self, model_dim, device='cpu', eps=1e-6):
        self.model_dim = model_dim
        self.device = device

        self.eps = eps

        self.gamma = torch.ones(self.model_dim, device=self.device)

    def forward(self, X):
        rms   = torch.sqrt((X * X).mean(dim=-1, keepdim=True) + self.eps)
        X_hat = X / rms

        # Cache for backward
        self.cache = RMSNorm.Grad_Info(x_hat=X_hat, rms=rms)

        return self.gamma * X_hat

    def backward(self, dY):
        D = self.model_dim
        cache = self.cache

        dX_hat      = dY * self.gamma
        self.dgamma = (dY * cache.x_hat).reshape(-1, D).sum(dim=0)    # (D,)

        # Same trick as LayerNorm backward, minus the mean-subtraction term
        dX = (1.0 / cache.rms) * (
            dX_hat
            - cache.x_hat * (dX_hat * cache.x_hat).mean(dim=-1, keepdim=True)
        )

        return dX

    def update(self, lr):
        self.gamma -= lr * self.dgamma
