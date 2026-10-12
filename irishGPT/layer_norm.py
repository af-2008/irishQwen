import torch
from collections import namedtuple

class LayerNorm:
    Grad_Info = namedtuple('Grad_Info', [
        'x',           # (B, T, C) original input (before LN)
        'x_hat',       # (B, T, C) normalized: (X - mu) / sigma
        'x_norm',      # (B, T, C) after gamma/beta scaling
        'sigma',       # (B, T, 1) std dev (for LN backward)
    ])

    def __init__(self, model_dim, device='cpu'):
        self.model_dim = model_dim
        self.device = device

        self.eps = 1e-5

        self.gamma = torch.ones(self.model_dim, device=self.device)
        self.beta  = torch.zeros(self.model_dim, device=self.device)

    def forward(self, X):
        mu    = X.mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(X.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        X_hat  = (X - mu) / sigma
        X_norm = self.gamma * X_hat + self.beta

        # Cache for backward
        self.cache = LayerNorm.Grad_Info(
            x=X, x_hat=X_hat, x_norm=X_norm, sigma=sigma,
        )

        return X_norm

    def backward(self, dX):
        B, T, C = dX.shape
        D = C   # feature dimension for LayerNorm
        cache = self.cache
        pass

        dX_hat      = dX * self.gamma
        self.dgamma = (dX * cache.x_hat).sum(dim=(0, 1))    # (C,)
        self.dbeta  = dX.sum(dim=(0, 1))                     # (C,)

        dX_ln = (1.0 / (D * cache.sigma)) * (
            D * dX_hat
            - dX_hat.sum(dim=-1, keepdim=True)
            - cache.x_hat * (dX_hat * cache.x_hat).sum(dim=-1, keepdim=True)
        )

        return dX_ln

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta
        pass