import math
import torch


class PositionalEmbedding:
    """
    Learned positional embeddings, shape (max_seq_len, embed_dim).
    """

    def __init__(self, max_seq_len, embed_dim, device="cpu"):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.device = device

        self.W = torch.randn(max_seq_len, embed_dim, device=self.device) * math.sqrt(2.0 / embed_dim)

    def forward(self, batch_size, seq_len):
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(seq_len, device=self.device, dtype=torch.long)
        self.positions = positions
        self.batch_size = batch_size

        pos = self.W[positions]  # (T, C)
        return pos.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def backward(self, dY):
        # dY shape: (B, T, C)
        self.dW = torch.zeros_like(self.W)
        per_pos = dY.sum(dim=0)  # (T, C), sum over batch
        self.dW.index_add_(0, self.positions, per_pos)

        # Positional input has no upstream parent in this model graph.
        return None

    def update(self, lr):
        self.W -= lr * self.dW
