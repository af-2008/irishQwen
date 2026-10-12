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
        # y = gamma * x / sqrt(mean(x^2) + eps)
        pass

        # Cache for backward

        return self.gamma * X_hat

    def backward(self, dY):
        return dX

    def update(self, lr):
        pass
